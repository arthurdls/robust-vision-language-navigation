#!/usr/bin/env python3
"""
Goal adherence experiment runner.

Runs goal adherence tasks with and without LLM diary monitoring.
For each task, runs multiple trials per condition and saves comprehensive results.

Control loop (simpler than LTL -- no subgoal decomposition):
  1. Teleport drone to initial_pos, reset OpenVLA model.
  2. Each step: capture frame, send instruction to OpenVLA, apply action poses.
  3. Without LLM (baseline):
     - Send the RAW subgoal string directly to OpenVLA as instruction
     - No conversion, no diary monitor
     - Run until convergence (small-change detection) or max_steps
  4. With LLM (diary-assisted):
     - Run SubgoalConverter to extract the OpenVLA instruction from the subgoal
     - Every diary_check_interval steps call LiveDiaryMonitor.on_frame():
       * stop  -> early stop (primary overshoot prevention)
       * override -> replace OpenVLA instruction until next checkpoint
       * continue -> keep going
     - On convergence: call LiveDiaryMonitor.on_convergence():
       * complete -> end run
       * stopped_short -> supervisor issues corrective command, resume
       * overshot -> supervisor issues reversal command, resume
       * correction cycle repeats up to max_corrections times

Usage (from repo root):
  python scripts/run_goal_adherence.py --task turn_right_until_red_car.json
  python scripts/run_goal_adherence.py --run-all
  python scripts/run_goal_adherence.py --task turn_right_until_red_car.json --model gpt-5
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
from typing import Any, Dict, List, Optional

from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sim_common import (
    BATCH_SCRIPT,
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
    set_drone_cam_and_get_image,
    setup_env_and_imports,
    setup_sim_env,
    state_for_openvla,
)

_AI_SRC = str(REPO_ROOT / "ai_framework" / "src")
if _AI_SRC not in sys.path:
    sys.path.insert(0, _AI_SRC)

from modules.diary_monitor import LiveDiaryMonitor
from modules.subgoal_converter import SubgoalConverter

GA_TASKS_DIR = REPO_ROOT / "tasks" / "goal_adherence_tasks"
GA_RESULTS_DIR = REPO_ROOT / "results" / "goal_adherence_results"

RUNS_PER_CONDITION = 3
SMALL_DELTA_POS = 3.0
SMALL_DELTA_YAW = 1.0
DEFAULT_MAX_CORRECTIONS = 10

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def _load_ga_task(path: Path) -> Dict[str, Any]:
    """Load a goal adherence task JSON and validate required fields."""
    with open(path, "r") as f:
        data = json.load(f)
    required = ["task_name", "subgoal", "initial_pos", "max_steps", "diary_check_interval"]
    for key in required:
        if key not in data:
            raise ValueError(f"Task JSON missing required field: {key}")
    return data


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def _run_single_ga(
    env: Any,
    task: Dict[str, Any],
    batch: Any,
    server_url: str,
    run_dir: Path,
    use_llm: bool,
    model: str,
    drone_cam_id: int,
    max_corrections: int = DEFAULT_MAX_CORRECTIONS,
) -> Dict[str, Any]:
    """Run one goal adherence trial. Returns run_info dict."""
    subgoal = task["subgoal"]
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps = task["max_steps"]
    check_interval = task["diary_check_interval"]

    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3]
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[4] - 180
    )
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)
    batch.reset_model(server_url)

    cam_id = drone_cam_id

    # --- Determine instruction and set up monitor ---
    monitor: Optional[LiveDiaryMonitor] = None
    converted_instruction: Optional[str] = None

    if use_llm:
        converter = SubgoalConverter(model=model)
        converted_instruction = converter.convert(subgoal)
        current_instruction = converted_instruction

        diary_artifacts = run_dir / "diary_artifacts"
        diary_artifacts.mkdir(parents=True, exist_ok=True)
        monitor = LiveDiaryMonitor(
            subgoal=subgoal,
            check_interval=check_interval,
            model=model,
            artifacts_dir=diary_artifacts,
            max_corrections=max_corrections,
        )
    else:
        current_instruction = subgoal

    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    last_pose: Optional[List[float]] = None
    small_count = 0
    trajectory_log: List[Dict[str, Any]] = []
    override_history: List[Dict[str, Any]] = []
    stop_reason = "max_steps"
    total_steps = 0
    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[4]
    start_ts = datetime.now().isoformat()

    step = 0
    while step < max_steps:
        batch.set_cam(env)
        image = set_drone_cam_and_get_image(env, cam_id)
        if image is None:
            logger.warning("No image at step %d, ending run.", step)
            stop_reason = "no_image"
            break

        frame_path = frames_dir / f"frame_{step:06d}.png"
        try:
            import cv2
            cv2.imwrite(str(frame_path), image)
        except Exception as e:
            logger.debug("Failed to save frame %s: %s", frame_path, e)

        # --- Diary monitor checkpoint (LLM runs only) ---
        if monitor is not None:
            result = monitor.on_frame(frame_path)
            if result.action == "stop":
                logger.info("LLM early stop at step %d: %s", step, result.reasoning)
                stop_reason = "llm_stopped"
                total_steps = step
                break
            if result.action == "override" and result.new_instruction:
                logger.info(
                    "LLM override at step %d: '%s' -> '%s'",
                    step, current_instruction, result.new_instruction,
                )
                override_history.append({
                    "step": step,
                    "old_instruction": current_instruction,
                    "new_instruction": result.new_instruction,
                    "reasoning": result.reasoning,
                })
                current_instruction = result.new_instruction

        # --- Send instruction to OpenVLA ---
        response = batch.send_prediction_request(
            image=Image.fromarray(image),
            proprio=state_for_openvla(current_pose),
            instr=current_instruction.strip().lower(),
            server_url=server_url,
        )

        if response is None:
            logger.warning("No VLA response at step %d, ending run.", step)
            stop_reason = "no_response"
            total_steps = step
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            logger.warning("Empty action at step %d, ending run.", step)
            stop_reason = "empty_action"
            total_steps = step
            break

        try:
            new_image, current_pose, steps_added = apply_action_poses(
                env,
                action_poses,
                origin_x,
                origin_y,
                origin_z,
                origin_yaw,
                batch.set_cam,
                trajectory_log=trajectory_log,
                sleep_s=0.1,
                drone_cam_id=cam_id,
            )
        except Exception as e:
            logger.error("Error executing action at step %d: %s", step, e)
            stop_reason = "action_error"
            total_steps = step
            break

        total_steps = step + 1

        # --- Convergence detection ---
        converged = False
        if last_pose is not None:
            diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
            if (
                all(d < SMALL_DELTA_POS for d in diffs[:3])
                and diffs[3] < SMALL_DELTA_YAW
            ):
                small_count += 1
            else:
                small_count = 0
            if small_count >= batch.ACTION_SMALL_STEPS:
                converged = True
        last_pose = list(current_pose)

        if converged:
            if monitor is not None:
                # Supervisor mode: evaluate convergence
                conv_frame = set_drone_cam_and_get_image(env, cam_id)
                if conv_frame is not None:
                    conv_path = frames_dir / f"frame_conv_{step:06d}.png"
                    try:
                        import cv2
                        cv2.imwrite(str(conv_path), conv_frame)
                    except Exception:
                        conv_path = frame_path
                else:
                    conv_path = frame_path

                conv_result = monitor.on_convergence(conv_path)

                if conv_result.action == "stop":
                    logger.info(
                        "Subgoal complete on convergence at step %d: %s",
                        step, conv_result.reasoning,
                    )
                    stop_reason = "llm_stopped"
                    break

                if conv_result.action == "command" and conv_result.new_instruction:
                    logger.info(
                        "Supervisor command at step %d: '%s'",
                        step, conv_result.new_instruction,
                    )
                    override_history.append({
                        "step": step,
                        "type": "supervisor_command",
                        "old_instruction": current_instruction,
                        "new_instruction": conv_result.new_instruction,
                        "reasoning": conv_result.reasoning,
                    })
                    current_instruction = conv_result.new_instruction
                    small_count = 0
                    last_pose = None
                    batch.reset_model(server_url)
                else:
                    logger.info("Convergence at step %d (supervisor gave no command).", step)
                    stop_reason = "convergence"
                    break
            else:
                logger.info("Convergence at step %d (no LLM).", step)
                stop_reason = "convergence"
                break

        step += 1
    else:
        stop_reason = "max_steps"
        total_steps = max_steps

    end_ts = datetime.now().isoformat()

    with open(run_dir / "trajectory_log.json", "w") as f:
        json.dump(trajectory_log, f, indent=2)

    run_info: Dict[str, Any] = {
        "task_config": task,
        "mode": "llm" if use_llm else "no_llm",
        "model": model if use_llm else None,
        "instruction_sent": converted_instruction if use_llm else subgoal,
        "total_steps": total_steps,
        "stop_reason": stop_reason,
        "instruction_overrides": override_history,
        "corrections_used": monitor.corrections_used if monitor else 0,
        "start_time": start_ts,
        "end_time": end_ts,
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    if monitor is not None:
        diary_summary = {
            "diary": monitor.diary,
            "override_history": override_history,
            "corrections_used": monitor.corrections_used,
            "stop_reason": stop_reason,
            "total_steps": total_steps,
        }
        with open(run_dir / "diary_summary.json", "w") as f:
            json.dump(diary_summary, f, indent=2)

    return run_info


# ---------------------------------------------------------------------------
# Multi-run orchestration
# ---------------------------------------------------------------------------

def _run_task_experiments(
    env: Any,
    task: Dict[str, Any],
    batch: Any,
    server_url: str,
    results_base: Path,
    model: str,
    drone_cam_id: int,
    runs_per_condition: int,
    max_corrections: int,
) -> None:
    """Run all experiments (no_llm + llm) for one task."""
    task_name = task["task_name"]
    task_dir = results_base / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    for run_idx in range(1, runs_per_condition + 1):
        run_name = f"no_llm_run_{run_idx:02d}"
        run_dir = task_dir / run_name
        logger.info("Running %s / %s", task_name, run_name)
        info = _run_single_ga(
            env, task, batch, server_url, run_dir,
            use_llm=False, model=model, drone_cam_id=drone_cam_id,
            max_corrections=max_corrections,
        )
        logger.info(
            "  %s: %d steps, stop_reason=%s",
            run_name, info["total_steps"], info["stop_reason"],
        )

    for run_idx in range(1, runs_per_condition + 1):
        run_name = f"llm_run_{run_idx:02d}"
        run_dir = task_dir / run_name
        logger.info("Running %s / %s", task_name, run_name)
        info = _run_single_ga(
            env, task, batch, server_url, run_dir,
            use_llm=True, model=model, drone_cam_id=drone_cam_id,
            max_corrections=max_corrections,
        )
        logger.info(
            "  %s: %d steps, stop_reason=%s",
            run_name, info["total_steps"], info["stop_reason"],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Run goal adherence experiments (with and without LLM diary monitoring)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--task",
        type=str,
        metavar="TASK.json",
        help="Run single task from tasks/goal_adherence_tasks/",
    )
    mode.add_argument(
        "--run-all",
        action="store_true",
        help="Run all JSON tasks in tasks/goal_adherence_tasks/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="VLM model for diary monitoring and subgoal conversion (default: gpt-4o).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_CONDITION,
        help=f"Number of runs per condition (default: {RUNS_PER_CONDITION}).",
    )
    parser.add_argument(
        "--max-corrections",
        type=int,
        default=DEFAULT_MAX_CORRECTIONS,
        help=f"Max supervisor corrections per LLM run (default: {DEFAULT_MAX_CORRECTIONS}).",
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
        "-p",
        "--server_port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="OpenVLA server port",
    )
    parser.add_argument(
        "-o",
        "--results_dir",
        default=str(GA_RESULTS_DIR),
        help="Base directory for results (default: results/goal_adherence_results)",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--use-default-cam",
        action="store_true",
        help="Use the default drone camera (ID %d) and skip interactive camera selection." % DRONE_CAM_ID,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    tasks: List[Dict[str, Any]] = []
    if args.task:
        path = Path(args.task)
        if not path.is_absolute():
            path = GA_TASKS_DIR / path.name
        if not path.exists():
            raise SystemExit(f"Task file not found: {path}")
        tasks.append(_load_ga_task(path))
    else:
        GA_TASKS_DIR.mkdir(parents=True, exist_ok=True)
        json_files = sorted(glob.glob(str(GA_TASKS_DIR / "*.json")))
        if not json_files:
            raise SystemExit(f"No JSON files found in {GA_TASKS_DIR}")
        for jf in json_files:
            try:
                tasks.append(_load_ga_task(Path(jf)))
            except Exception as e:
                logger.warning("Skipping %s: %s", jf, e)

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    setup_env_and_imports()
    batch = import_batch_module()
    os.chdir(str(UAV_FLOW_EVAL))

    server_url = f"http://127.0.0.1:{args.server_port}/predict"
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    env = setup_sim_env(args.env_id, int(args.time_dilation), int(args.seed), batch)

    drone_cam_id = DRONE_CAM_ID
    if not args.use_default_cam:
        logger.info("Camera selection: pick the camera for OpenVLA.")
        initial_pos_for_cam = tasks[0]["initial_pos"] if tasks else [0, 0, 0, 0]
        drone_cam_id = interactive_camera_select(env, initial_pos_for_cam, batch)

    for idx, task in enumerate(tasks):
        logger.info(
            "\n===== Task %d/%d: %s =====",
            idx + 1, len(tasks), task["task_name"],
        )
        try:
            _run_task_experiments(
                env, task, batch, server_url, results_base,
                args.model, drone_cam_id, args.runs,
                args.max_corrections,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted during task %s.", task["task_name"])
            break
        logger.info("===== Task %s finished =====\n", task["task_name"])

    env.close()


if __name__ == "__main__":
    main()
