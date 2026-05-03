#!/usr/bin/env python3
"""
Condition 3: Open-Loop LTL (No Monitor, No Corrections).

Uses the full LTL planner and SubgoalConverter, but runs the OpenVLA control
loop without any VLM monitoring. When the drone stops moving (convergence),
assumes the subgoal is complete and advances the automaton.

This is the most important baseline: it directly tests whether the low-level
controller executes subgoals reliably without monitoring.

OpenVLA server must be running: python scripts/start_server.py

Usage (from repo root):
  python scripts/run_condition3_open_loop.py --task third_task.json
  python scripts/run_condition3_open_loop.py --run_all_tasks
  python scripts/run_condition3_open_loop.py -c "Go to the tree then land" --initial-position -181,7331,876,-89
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
    DEFAULT_LLM_MODEL,
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
    relative_pose_to_world,
    set_drone_cam_and_get_image,
    setup_sim_env,
    state_for_openvla,
)

CONDITION3_TASKS_DIR = REPO_ROOT / "tasks" / "condition3"
CONDITION3_RESULTS_DIR = REPO_ROOT / "results" / "condition3"

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
        "max_steps_per_subgoal": int(data.get("max_steps_per_subgoal", DEFAULT_MAX_STEPS_PER_SUBGOAL)),
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
            "max_steps_per_subgoal": args.max_steps_per_subgoal,
        }]

    if task_file is not None:
        path = Path(task_file)
        if not path.is_absolute():
            path = CONDITION3_TASKS_DIR / path.name
        if not path.exists():
            raise SystemExit(f"Task file not found: {path}")
        task = _load_task(path)
        if args.max_steps_per_subgoal:
            task["max_steps_per_subgoal"] = args.max_steps_per_subgoal
        return [task]

    CONDITION3_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(CONDITION3_TASKS_DIR / "*.json")))
    if not json_files:
        raise SystemExit(f"No JSON files found in {CONDITION3_TASKS_DIR}")
    tasks = []
    for jf in json_files:
        try:
            task = _load_task(Path(jf))
            if args.max_steps_per_subgoal:
                task["max_steps_per_subgoal"] = args.max_steps_per_subgoal
            tasks.append(task)
        except Exception as e:
            logger.warning("Skipping %s: %s", jf, e)
    return tasks


def _sanitize_name(text: str, max_len: int = 40) -> str:
    clean = text.lower().replace(" ", "_")
    safe = "".join(c for c in clean if c.isalnum() or c == "_")
    return safe[:max_len] or "subgoal"


def run_open_loop_control_loop(
    env, batch, task, server_url, run_dir,
    llm_model, converter_model, drone_cam_id,
    save_mp4=False, mp4_fps=10.0,
    seed=DEFAULT_SEED, time_dilation=DEFAULT_TIME_DILATION, env_id="",
):
    from rvln.ai.llm_interface import LLMUserInterface
    from rvln.ai.ltl_planner import LTLSymbolicPlanner
    from rvln.ai.subgoal_converter import SubgoalConverter

    instruction = task["instruction"]
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps_per_subgoal = task["max_steps_per_subgoal"]

    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env.teleport(initial_pos[0:3], initial_pos[4])
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    start_ts = datetime.now().isoformat()

    logger.info("Planning instruction: '%s'", instruction)
    llm_interface = LLMUserInterface(model=llm_model)
    planner = LTLSymbolicPlanner(llm_interface)
    planner.plan_from_natural_language(instruction)

    ltl_plan = {
        "ltl_nl_formula": llm_interface.ltl_nl_formula.get("ltl_nl_formula", ""),
        "pi_predicates": dict(planner.pi_map),
    }

    subgoal_summaries = []
    trajectory_log = []
    total_frame_count = 0
    subgoal_index = 0

    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[4]

    current_subgoal = planner.get_next_predicate()
    if current_subgoal is None:
        raise RuntimeError("LTL planning produced no subgoals.")

    converter = SubgoalConverter(model=converter_model)

    while current_subgoal is not None:
        subgoal_index += 1
        safe_name = _sanitize_name(current_subgoal)
        subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"
        subgoal_dir.mkdir(parents=True, exist_ok=True)

        conversion = converter.convert(current_subgoal)
        converted_instruction = conversion.instruction

        logger.info("--- Subgoal %d: '%s' -> '%s' ---",
                     subgoal_index, current_subgoal, converted_instruction)

        batch.reset_model(server_url)
        current_pose = [0.0, 0.0, 0.0, 0.0]
        last_pose = None
        small_count = 0
        stop_reason = "max_steps"
        total_steps = 0

        for step in range(max_steps_per_subgoal):
            image = set_drone_cam_and_get_image(env, drone_cam_id)
            if image is None:
                stop_reason = "no_image"
                total_steps = step
                break

            global_frame_idx = total_frame_count + step
            frame_path = frames_dir / f"frame_{global_frame_idx:06d}.png"
            try:
                import cv2
                cv2.imwrite(str(frame_path), image)
            except Exception:
                pass

            response = batch.send_prediction_request(
                image=Image.fromarray(image),
                proprio=state_for_openvla(current_pose),
                instr=converted_instruction.strip().lower(),
                server_url=server_url,
            )

            if response is None:
                stop_reason = "no_response"
                total_steps = step
                break

            action_poses = response.get("action")
            if not isinstance(action_poses, list) or len(action_poses) == 0:
                stop_reason = "empty_action"
                total_steps = step
                break

            try:
                new_image, current_pose, steps_added = apply_action_poses(
                    env, action_poses, origin_x, origin_y, origin_z, origin_yaw,
                    trajectory_log=trajectory_log, sleep_s=0.1, drone_cam_id=drone_cam_id,
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
                    logger.info("Convergence at step %d, advancing subgoal.", step)
                    stop_reason = "convergence"
                    break
            last_pose = list(current_pose)

            if response.get("done") is True:
                logger.info("Model reported done at step %d.", step)
                stop_reason = "model_done"
                break
        else:
            stop_reason = "max_steps"
            total_steps = max_steps_per_subgoal

        next_origin_x, next_origin_y, next_origin_z, next_origin_yaw = relative_pose_to_world(
            origin_x, origin_y, origin_z, origin_yaw, current_pose,
        )

        total_frame_count += total_steps
        subgoal_summaries.append({
            "subgoal": current_subgoal,
            "converted_instruction": converted_instruction,
            "total_steps": total_steps,
            "stop_reason": stop_reason,
            "corrections_used": 0,
            "vlm_call_count": 0,
            "vlm_call_records": list(converter.llm_call_records),
            "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
        })
        converter.llm_call_records.clear()

        origin_x, origin_y, origin_z, origin_yaw = (
            next_origin_x, next_origin_y, next_origin_z, next_origin_yaw,
        )
        planner.advance_state(current_subgoal)
        current_subgoal = planner.get_next_predicate()

    end_ts = datetime.now().isoformat()

    with open(run_dir / "trajectory_log.json", "w") as f:
        json.dump(trajectory_log, f, indent=2)

    playback_mp4 = None
    if save_mp4:
        try:
            from rvln.eval.playback import save_run_directory_mp4
            playback_mp4 = save_run_directory_mp4(run_dir, fps=mp4_fps)
        except Exception as e:
            logger.warning("Could not write playback.mp4: %s", e)

    all_vlm_records = []
    for s in subgoal_summaries:
        all_vlm_records.extend(s.get("vlm_call_records", []))
    total_input_tokens = sum(r.get("input_tokens", 0) for r in all_vlm_records)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in all_vlm_records)
    end_dt = datetime.fromisoformat(end_ts)
    start_dt = datetime.fromisoformat(start_ts)
    wall_clock_seconds = (end_dt - start_dt).total_seconds()

    run_info = {
        "condition": "condition3_open_loop",
        "task": task,
        "seed": seed,
        "time_dilation": time_dilation,
        "env_id": env_id,
        "server_url": server_url,
        "drone_cam_id": drone_cam_id,
        "llm_model": llm_model,
        "converter_model": converter_model,
        "models": {
            "ltl_nl_planning": llm_model,
            "subgoal_converter": converter_model,
            "openvla_predict_url": server_url,
        },
        "config": {
            "max_steps_per_subgoal": max_steps_per_subgoal,
            "convergence_thresholds": {
                "position_delta": ACTION_SMALL_DELTA_POS,
                "yaw_delta": ACTION_SMALL_DELTA_YAW,
                "consecutive_steps": batch.ACTION_SMALL_STEPS,
            },
        },
        "ltl_plan": ltl_plan,
        "subgoal_count": subgoal_index,
        "subgoal_summaries": subgoal_summaries,
        "total_steps": sum(s["total_steps"] for s in subgoal_summaries),
        "total_vlm_calls": 0,
        "total_corrections": 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "vlm_call_records": all_vlm_records,
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
        description="Condition 3: Open-loop LTL (no monitor, no corrections)",
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
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--converter_model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--max-steps-per-subgoal", type=int, default=DEFAULT_MAX_STEPS_PER_SUBGOAL)
    parser.add_argument("-o", "--results_dir", default=str(CONDITION3_RESULTS_DIR))
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
            run_name = f"c3_open_loop__{task_label}__{ts}"
            try:
                run_dir = results_base / run_name
                run_info = run_open_loop_control_loop(
                    env=env, batch=batch, task=task, server_url=server_url,
                    run_dir=run_dir, llm_model=args.llm_model,
                    converter_model=args.converter_model,
                    drone_cam_id=drone_cam_id, save_mp4=args.save_mp4, mp4_fps=args.mp4_fps,
                    seed=args.seed, time_dilation=args.time_dilation,
                    env_id=args.env_id,
                )
                logger.info("Run saved to %s (%d subgoals, %d total steps)",
                            run_dir, run_info["subgoal_count"], run_info["total_steps"])
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
