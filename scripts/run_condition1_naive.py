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
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from PIL import Image

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.config import (
    ACTION_SMALL_DELTA_POS,
    ACTION_SMALL_DELTA_YAW,
    DEFAULT_MAX_STEPS_PER_SUBGOAL,
    DEFAULT_SEED,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SIM_API_PORT,
    DEFAULT_SIM_HOST,
    DEFAULT_SIM_PORT,
    DEFAULT_TIME_DILATION,
)
from rvln.eval.task_utils import get_completed_task_ids, resolve_eval_tasks, sanitize_run_label
from rvln.paths import (
    BATCH_SCRIPT,
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

SHARED_TASKS_DIR = REPO_ROOT / "tasks"
CONDITION1_RESULTS_DIR = REPO_ROOT / "results" / "condition1"

logger = logging.getLogger(__name__)


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
    save_frames: bool = True,
) -> Dict[str, Any]:
    instruction = task["instruction"]
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps = task["max_steps"]

    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env.teleport(initial_pos[0:3], initial_pos[3])
    time.sleep(batch.SLEEP_AFTER_RESET_S)
    batch.reset_model(server_url)

    cam_id = drone_cam_id
    start_ts = datetime.now().isoformat()

    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[3]
    last_pose: Optional[List[float]] = None
    small_count = 0
    trajectory_log: List[Dict[str, Any]] = []
    stop_reason = "max_steps"
    total_steps = 0

    from rvln.eval.step_timer import StepTimer
    from rvln.eval.async_frame_writer import AsyncFrameWriter
    _step_timer = StepTimer(run_dir / "step_timings.jsonl")
    _frame_writer = AsyncFrameWriter(frames_dir, enabled=save_frames)

    # Loop-carried frame: first iteration fetches via /get_frame, subsequent
    # iterations reuse the image returned by /step.
    image: Optional[np.ndarray] = None

    for step in range(max_steps):
        _step_timer.start_step(step)
        try:
            if image is None:
                with _step_timer.phase("get_frame"):
                    image = set_drone_cam_and_get_image(env, cam_id)
                if image is None:
                    logger.warning("No image at step %d, ending run.", step)
                    stop_reason = "no_image"
                    total_steps = step
                    break

            frame_path = frames_dir / f"frame_{step:06d}.png"
            with _step_timer.phase("frame_write"):
                _frame_writer.write(f"frame_{step:06d}.png", image)

            with _step_timer.phase("predict"):
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

            with _step_timer.phase("apply_action"):
                try:
                    new_image, current_pose, steps_added = apply_action_poses(
                        env,
                        action_poses,
                        origin_x,
                        origin_y,
                        origin_z,
                        origin_yaw,
                        trajectory_log=trajectory_log,
                        drone_cam_id=cam_id,
                    )
                except Exception as e:
                    logger.error("Error executing action at step %d: %s", step, e)
                    stop_reason = "action_error"
                    total_steps = step
                    break

            total_steps = step + 1

            image = new_image  # None triggers /get_frame fallback on next iteration.

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
        finally:
            _step_timer.end_step()
    else:
        stop_reason = "max_steps"
        total_steps = max_steps

    _step_timer.close()
    _frame_writer.close()

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
        "completed": True,
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

    parser.add_argument("--initial-position", type=str, default=None, metavar="x,y,z,yaw")
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
    parser.add_argument("--no-save-frames", action="store_true",
                        help="Skip per-step PNG writes (faster benchmarks)")
    parser.add_argument("--select-cam", action="store_true",
                        help="Interactively pick the drone camera instead of auto-detecting")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

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
    tasks = resolve_eval_tasks(args, map_info, SHARED_TASKS_DIR, overrides={"max_steps": "max_steps"})

    try:
        drone_cam_id = env.drone_cam_id
        if args.select_cam:
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
            logger.info("\n===== Task %d/%d: '%s' =====", idx + 1, len(tasks), task["instruction"][:80])
            ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            task_label = task.get("task_id") or sanitize_run_label(task["instruction"], max_len=30)
            run_name = f"c1_naive__{task_label}__{ts}"
            try:
                run_dir = results_base / run_name
                run_info = run_naive_control_loop(
                    env=env, batch=batch, task=task, server_url=server_url,
                    run_dir=run_dir, drone_cam_id=drone_cam_id,
                    save_mp4=args.save_mp4, mp4_fps=args.mp4_fps,
                    seed=args.seed, time_dilation=args.time_dilation,
                    env_id=map_info.env_id,
                    save_frames=not args.no_save_frames,
                )
                logger.info("Run saved to %s (%d steps, stop=%s)",
                            run_dir, run_info["total_steps"], run_info["stop_reason"])
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
