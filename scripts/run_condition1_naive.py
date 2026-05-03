#!/usr/bin/env python3
"""
Condition 1: Naive End-to-End VLA.

Passes the full multi-step NL instruction directly to OpenVLA as a single
instruction. No LTL decomposition, no subgoal conversion, no goal adherence monitoring,
no convergence corrections.

The drone runs until convergence (stops moving) or max_steps. This establishes
that the base VLA cannot handle long-horizon, multi-step instructions.

OpenVLA server must be running: python scripts/start_server.py

Usage (from repo root):
  python scripts/run_condition1_naive.py --task third_task.json
  python scripts/run_condition1_naive.py --run_all_tasks
  python scripts/run_condition1_naive.py -c "Go to the tree then the streetlight" --initial-position -181,7331,876,-89
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

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.config import (
    ACTION_SMALL_DELTA_POS,
    ACTION_SMALL_DELTA_YAW,
    DEFAULT_INITIAL_POSITION,
    DEFAULT_MAX_STEPS_PER_SUBGOAL,
    DEFAULT_SEED,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SIM_API_PORT,
    DEFAULT_SIM_HOST,
    DEFAULT_SIM_PORT,
    DEFAULT_TIME_DILATION,
)
from rvln.paths import (
    BATCH_SCRIPT,
    DOWNTOWN_ENV_ID,
    REPO_ROOT,
    UAV_FLOW_EVAL,
)
from rvln.sim.env_setup import (
    apply_action_poses,
    import_batch_module,
    interactive_camera_select,
    load_env_vars,
    normalize_initial_pos,
    parse_position,
    set_drone_cam_and_get_image,
    setup_sim_env,
    state_for_openvla,
)

CONDITION1_TASKS_DIR = REPO_ROOT / "tasks" / "condition1"
CONDITION1_RESULTS_DIR = REPO_ROOT / "results" / "condition1"

logger = logging.getLogger(__name__)


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


def _load_task(path: Path) -> Dict[str, Any]:
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
    }
    for passthrough_key in ("task_id", "category", "difficulty", "region", "notes"):
        if passthrough_key in data:
            result[passthrough_key] = data[passthrough_key]
    return result


def _resolve_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
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

    if cmd is not None:
        initial_pos_str = getattr(args, "initial_position", None) or DEFAULT_INITIAL_POSITION
        return [{
            "instruction": cmd.strip(),
            "initial_pos": parse_position(initial_pos_str),
            "max_steps": args.max_steps,
        }]

    if task_file is not None:
        path = Path(task_file)
        if not path.is_absolute():
            path = CONDITION1_TASKS_DIR / path.name
        if not path.exists():
            raise SystemExit(f"Task file not found: {path}")
        task = _load_task(path)
        if args.max_steps:
            task["max_steps"] = args.max_steps
        return [task]

    CONDITION1_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(CONDITION1_TASKS_DIR / "*.json")))
    if not json_files:
        raise SystemExit(f"No JSON files found in {CONDITION1_TASKS_DIR}")
    tasks = []
    for jf in json_files:
        try:
            task = _load_task(Path(jf))
            if args.max_steps:
                task["max_steps"] = args.max_steps
            tasks.append(task)
        except Exception as e:
            logger.warning("Skipping %s: %s", jf, e)
    return tasks


def run_naive_control_loop(
    env: Any,
    batch: Any,
    task: Dict[str, Any],
    server_url: str,
    run_dir: Path,
    drone_cam_id: int,
    save_mp4: bool = False,
    mp4_fps: float = 10.0,
    seed: int = DEFAULT_SEED,
    time_dilation: int = DEFAULT_TIME_DILATION,
    env_id: str = "",
) -> Dict[str, Any]:
    instruction = task["instruction"]
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps = task["max_steps"]

    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env.teleport(initial_pos[0:3], initial_pos[4])
    time.sleep(batch.SLEEP_AFTER_RESET_S)
    batch.reset_model(server_url)

    cam_id = drone_cam_id
    start_ts = datetime.now().isoformat()

    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[4]
    last_pose: Optional[List[float]] = None
    small_count = 0
    trajectory_log: List[Dict[str, Any]] = []
    stop_reason = "max_steps"
    total_steps = 0

    for step in range(max_steps):
        image = set_drone_cam_and_get_image(env, cam_id)
        if image is None:
            logger.warning("No image at step %d, ending run.", step)
            stop_reason = "no_image"
            total_steps = step
            break

        frame_path = frames_dir / f"frame_{step:06d}.png"
        try:
            import cv2
            cv2.imwrite(str(frame_path), image)
        except Exception as e:
            logger.debug("Failed to save frame %s: %s", frame_path, e)

        response = batch.send_prediction_request(
            image=Image.fromarray(image),
            proprio=state_for_openvla(current_pose),
            instr=instruction.strip().lower(),
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

        if last_pose is not None:
            diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
            if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                small_count += 1
            else:
                small_count = 0
            if small_count >= batch.ACTION_SMALL_STEPS:
                logger.info("Convergence at step %d (drone stopped moving).", step)
                stop_reason = "convergence"
                break
        last_pose = list(current_pose)

        if response.get("done") is True:
            logger.info("Model reported done at step %d.", step)
            stop_reason = "model_done"
            break
    else:
        stop_reason = "max_steps"
        total_steps = max_steps

    end_ts = datetime.now().isoformat()

    with open(run_dir / "trajectory_log.json", "w") as f:
        json.dump(trajectory_log, f, indent=2)

    playback_mp4: Optional[Path] = None
    if save_mp4:
        try:
            from rvln.eval.playback import save_run_directory_mp4
            playback_mp4 = save_run_directory_mp4(run_dir, fps=mp4_fps)
        except Exception as e:
            logger.warning("Could not write playback.mp4: %s", e)

    end_dt = datetime.fromisoformat(end_ts)
    start_dt = datetime.fromisoformat(start_ts)
    wall_clock_seconds = (end_dt - start_dt).total_seconds()

    run_info: Dict[str, Any] = {
        "condition": "condition1_naive",
        "task": task,
        "seed": seed,
        "time_dilation": time_dilation,
        "env_id": env_id,
        "server_url": server_url,
        "drone_cam_id": drone_cam_id,
        "config": {
            "max_steps": max_steps,
            "convergence_thresholds": {
                "position_delta": ACTION_SMALL_DELTA_POS,
                "yaw_delta": ACTION_SMALL_DELTA_YAW,
                "consecutive_steps": batch.ACTION_SMALL_STEPS,
            },
        },
        "instruction_sent": instruction,
        "total_steps": total_steps,
        "stop_reason": stop_reason,
        "total_vlm_calls": 0,
        "total_corrections": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "playback_mp4": str(playback_mp4) if playback_mp4 else None,
        "start_time": start_ts,
        "end_time": end_ts,
        "wall_clock_seconds": round(wall_clock_seconds, 2),
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    return run_info


def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Condition 1: Naive end-to-end VLA (no decomposition, no monitoring)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-c", "--command", type=str, default=None)
    mode.add_argument("--task", type=str, default=None, metavar="TASK.json")
    mode.add_argument("--run_all_tasks", action="store_true")

    parser.add_argument("--initial-position", type=str, default=DEFAULT_INITIAL_POSITION, metavar="x,y,z,yaw")
    parser.add_argument("-e", "--env_id", default=DOWNTOWN_ENV_ID)
    parser.add_argument("-t", "--time_dilation", type=int, default=DEFAULT_TIME_DILATION)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-p", "--server_port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--server_host", type=str, default=DEFAULT_SERVER_HOST)
    parser.add_argument("--sim_host", type=str, default=DEFAULT_SIM_HOST)
    parser.add_argument("--sim_port", type=int, default=DEFAULT_SIM_PORT)
    parser.add_argument("--sim_api_port", type=int, default=DEFAULT_SIM_API_PORT)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS_PER_SUBGOAL)
    parser.add_argument("-o", "--results_dir", default=str(CONDITION1_RESULTS_DIR))
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--mp4-fps", type=float, default=10.0)
    parser.add_argument("--select-cam", action="store_true",
                        help="Interactively pick the drone camera instead of auto-detecting")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    tasks = _resolve_tasks(args)

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    batch = import_batch_module()
    os.chdir(str(UAV_FLOW_EVAL))

    server_url = f"http://{args.server_host}:{args.server_port}/predict"
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    env = setup_sim_env(args.env_id, int(args.time_dilation), int(args.seed), batch,
                        sim_host=args.sim_host, sim_api_port=args.sim_api_port)

    try:
        drone_cam_id = env.drone_cam_id
        if args.select_cam:
            initial_pos_for_cam = (
                tasks[0]["initial_pos"] if tasks
                else normalize_initial_pos(parse_position(DEFAULT_INITIAL_POSITION))
            )
            drone_cam_id = interactive_camera_select(env, initial_pos_for_cam, batch)

        def _sanitize_name(text: str, max_len: int = 40) -> str:
            clean = text.lower().replace(" ", "_")
            safe = "".join(c for c in clean if c.isalnum() or c == "_")
            return safe[:max_len] or "task"

        completed_ids = _get_completed_task_ids(results_base)
        if completed_ids:
            logger.info("Found %d completed task(s) in %s", len(completed_ids), results_base)

        for idx, task in enumerate(tasks):
            task_id = task.get("task_id", "")
            if task_id and task_id in completed_ids:
                logger.info("Skipping already completed task: %s", task_id)
                continue
            logger.info("\n===== Task %d/%d: '%s' =====", idx + 1, len(tasks), task["instruction"][:80])
            ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            task_label = task.get("task_id") or _sanitize_name(task["instruction"], max_len=30)
            run_name = f"c1_naive__{task_label}__{ts}"
            try:
                run_dir = results_base / run_name
                run_info = run_naive_control_loop(
                    env=env, batch=batch, task=task, server_url=server_url,
                    run_dir=run_dir, drone_cam_id=drone_cam_id,
                    save_mp4=args.save_mp4, mp4_fps=args.mp4_fps,
                    seed=args.seed, time_dilation=args.time_dilation,
                    env_id=args.env_id,
                )
                logger.info("Run saved to %s (%d steps, stop=%s)",
                            run_dir, run_info["total_steps"], run_info["stop_reason"])
            except KeyboardInterrupt:
                logger.info("Task interrupted.")
                break
            except Exception as e:
                logger.error("Task failed: %s", e, exc_info=True)
            logger.info("===== Task %d finished =====\n", idx + 1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
