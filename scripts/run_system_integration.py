#!/usr/bin/env python3
"""
Integrated LTL planner + LiveDiaryMonitor control loop.

Combines the LTL-NL neuro-symbolic planner (multi-step instruction decomposition
via Spot automaton) with the LiveDiaryMonitor (diary-based subgoal supervision
with convergence corrections).

For each subgoal produced by the planner:
  1. SubgoalConverter rewrites the NL predicate into a short OpenVLA instruction.
  2. A fresh LiveDiaryMonitor supervises execution with periodic VLM checkpoints,
     completion tracking, and corrective commands on convergence.
  3. When the monitor confirms completion (or the step budget is exhausted),
     the planner advances to the next subgoal.

This replaces the simple GoalAdherenceMonitor used in run_ltl_planner.py with
the full diary-based supervision pipeline from run_goal_adherence.py.

OpenVLA server must be running: python scripts/start_openvla_server.py

Usage (from repo root):
  # Single task from tasks/system_tasks/
  python scripts/run_system_integration.py --task third_task.json
  # All tasks in tasks/system_tasks/
  python scripts/run_system_integration.py --run_all_tasks
  # Ad-hoc command
  python scripts/run_system_integration.py -c "Go to the tree then land" --initial-position -181,7331,876,-89
  # Custom models and diary parameters
  python scripts/run_system_integration.py --task first_task.json --llm_model gpt-4o --monitor_model gpt-5.4 \\
      --diary-check-interval 10 --max-steps-per-subgoal 200 --max-corrections 10
"""

import argparse
import glob
import json
import logging
import math
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

_AI_SRC = str(REPO_ROOT / "ai_framework" / "src")
if _AI_SRC not in sys.path:
    sys.path.insert(0, _AI_SRC)

SYSTEM_TASKS_DIR = REPO_ROOT / "tasks" / "system_tasks"
INTEGRATION_RESULTS_DIR = REPO_ROOT / "results" / "integration_results"

DEFAULT_MAX_STEPS_PER_SUBGOAL = 300
DEFAULT_DIARY_CHECK_INTERVAL = 10
DEFAULT_MAX_CORRECTIONS = 15
SMALL_DELTA_POS = 3.0
SMALL_DELTA_YAW = 1.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def _load_task(path: Path) -> Dict[str, Any]:
    """Load an LTL task JSON with optional diary parameters."""
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
    return {
        "instruction": instruction,
        "initial_pos": [float(x) for x in initial_pos],
        "max_steps_per_subgoal": int(data.get("max_steps_per_subgoal", DEFAULT_MAX_STEPS_PER_SUBGOAL)),
        "diary_check_interval": int(data.get("diary_check_interval", DEFAULT_DIARY_CHECK_INTERVAL)),
        "max_corrections": int(data.get("max_corrections", DEFAULT_MAX_CORRECTIONS)),
    }


def _resolve_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build task list from CLI arguments."""
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
            "diary_check_interval": args.diary_check_interval,
            "max_corrections": args.max_corrections,
        }]

    if task_file is not None:
        path = Path(task_file)
        if not path.is_absolute():
            path = SYSTEM_TASKS_DIR / path.name
        if not path.exists():
            raise SystemExit(f"Task file not found: {path}")
        task = _load_task(path)
        task["max_steps_per_subgoal"] = args.max_steps_per_subgoal or task["max_steps_per_subgoal"]
        task["diary_check_interval"] = args.diary_check_interval or task["diary_check_interval"]
        task["max_corrections"] = args.max_corrections or task["max_corrections"]
        return [task]

    SYSTEM_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(SYSTEM_TASKS_DIR / "*.json")))
    if not json_files:
        raise SystemExit(f"No JSON files found in {SYSTEM_TASKS_DIR}")
    tasks = []
    for jf in json_files:
        try:
            task = _load_task(Path(jf))
            task["max_steps_per_subgoal"] = args.max_steps_per_subgoal or task["max_steps_per_subgoal"]
            task["diary_check_interval"] = args.diary_check_interval or task["diary_check_interval"]
            task["max_corrections"] = args.max_corrections or task["max_corrections"]
            tasks.append(task)
        except Exception as e:
            logger.warning("Skipping %s: %s", jf, e)
    return tasks


def _sanitize_name(text: str, max_len: int = 40) -> str:
    """Create a filesystem-safe short name from text."""
    clean = text.lower().replace(" ", "_")
    safe = "".join(c for c in clean if c.isalnum() or c == "_")
    return safe[:max_len] or "subgoal"


# ---------------------------------------------------------------------------
# Per-subgoal control loop
# ---------------------------------------------------------------------------

def _run_subgoal(
    env: Any,
    batch: Any,
    server_url: str,
    subgoal_nl: str,
    monitor_model: str,
    check_interval: int,
    max_steps: int,
    max_corrections: int,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    origin_yaw: float,
    drone_cam_id: int,
    frames_dir: Path,
    subgoal_dir: Path,
    subgoal_index: int,
    frame_offset: int,
    trajectory_log: List[Dict[str, Any]],
    frame_metadata: List[Dict[str, Any]],
    telemetry_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the diary-monitored control loop for a single subgoal.

    Returns a dict with subgoal-level results and the final world-space origin
    for the next subgoal.
    """
    from modules.diary_monitor import DiaryCheckResult, LiveDiaryMonitor
    from modules.subgoal_converter import SubgoalConverter

    subgoal_dir.mkdir(parents=True, exist_ok=True)
    diary_artifacts = subgoal_dir / "diary_artifacts"
    diary_artifacts.mkdir(parents=True, exist_ok=True)

    converter = SubgoalConverter(model=monitor_model)
    convert_start = time.perf_counter()
    convert_error: Optional[str] = None
    try:
        converted_instruction = converter.convert(subgoal_nl)
    except Exception as e:
        convert_error = str(e)
        raise
    finally:
        convert_end = time.perf_counter()
        telemetry_log.append({
            "event_type": "vlm_inference",
            "query_type": "subgoal_conversion",
            "subgoal_index": subgoal_index,
            "subgoal": subgoal_nl,
            "model": monitor_model,
            "duration_ms": (convert_end - convert_start) * 1000.0,
            "success": convert_error is None,
            "error": convert_error,
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "timestamp_unix_s": time.time(),
        })
    current_instruction = converted_instruction

    monitor = LiveDiaryMonitor(
        subgoal=subgoal_nl,
        check_interval=check_interval,
        model=monitor_model,
        artifacts_dir=diary_artifacts,
        max_corrections=max_corrections,
    )

    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    openvla_pose_origin: List[float] = [0.0, 0.0, 0.0, 0.0]
    last_pose: Optional[List[float]] = None
    small_count = 0
    override_history: List[Dict[str, Any]] = []
    in_correction = False
    last_correction_step = -check_interval
    stop_reason = "max_steps"
    total_steps = 0

    batch.reset_model(server_url)

    cam_id = drone_cam_id
    result = None
    step = 0

    while step < max_steps:
        batch.set_cam(env)
        image = set_drone_cam_and_get_image(env, cam_id)
        frame_capture_time_unix_s = time.time()
        frame_capture_time_iso = datetime.now().isoformat(timespec="milliseconds")
        if image is None:
            logger.warning("No image at step %d, ending subgoal.", step)
            stop_reason = "no_image"
            break

        global_frame_idx = frame_offset + step
        frame_path = frames_dir / f"frame_{global_frame_idx:06d}.png"
        try:
            import cv2
            if cv2.imwrite(str(frame_path), image):
                frame_metadata.append({
                    "frame_file": frame_path.name,
                    "frame_index": global_frame_idx,
                    "frame_type": "step",
                    "subgoal_index": subgoal_index,
                    "subgoal": subgoal_nl,
                    "converted_instruction": converted_instruction,
                    "subgoal_step": step,
                    "captured_at_iso": frame_capture_time_iso,
                    "captured_at_unix_s": frame_capture_time_unix_s,
                })
        except Exception as e:
            logger.debug("Failed to save frame %s: %s", frame_path, e)

        monitor_start = time.perf_counter()
        monitor_error: Optional[str] = None
        try:
            result = monitor.on_frame(frame_path, displacement=list(current_pose))
        except Exception as e:
            monitor_error = str(e)
            logger.error("monitor.on_frame failed at step %d: %s", step, e)
            result = DiaryCheckResult(
                action="continue", new_instruction="", reasoning="",
                diary_entry="", completion_pct=monitor.last_completion_pct,
            )
        monitor_end = time.perf_counter()
        telemetry_log.append({
            "event_type": "vlm_inference",
            "query_type": "diary_on_frame",
            "subgoal_index": subgoal_index,
            "subgoal": subgoal_nl,
            "subgoal_step": step,
            "model": monitor_model,
            "duration_ms": (monitor_end - monitor_start) * 1000.0,
            "success": monitor_error is None,
            "error": monitor_error,
            "result_action": result.action if result is not None else None,
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "timestamp_unix_s": time.time(),
        })

        if result.action == "stop":
            logger.info("Monitor stop at step %d: %s", step, result.reasoning)
            stop_reason = "monitor_complete"
            total_steps = step
            break
        if result.action == "force_converge":
            logger.info("Monitor force_converge at step %d: %s", step, result.reasoning)
            override_history.append({
                "step": step,
                "type": "force_converge",
                "reasoning": result.reasoning,
            })

        openvla_pose = [c - o for c, o in zip(current_pose, openvla_pose_origin)]
        openvla_start = time.perf_counter()
        openvla_error: Optional[str] = None
        try:
            response = batch.send_prediction_request(
                image=Image.fromarray(image),
                proprio=state_for_openvla(openvla_pose),
                instr=current_instruction.strip().lower(),
                server_url=server_url,
            )
        except Exception as e:
            openvla_error = str(e)
            raise
        finally:
            openvla_end = time.perf_counter()
            telemetry_log.append({
                "event_type": "openvla_inference",
                "query_type": "predict",
                "subgoal_index": subgoal_index,
                "subgoal": subgoal_nl,
                "subgoal_step": step,
                "duration_ms": (openvla_end - openvla_start) * 1000.0,
                "success": openvla_error is None,
                "error": openvla_error,
                "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                "timestamp_unix_s": time.time(),
            })

        if response is None:
            logger.warning("No VLA response at step %d, ending subgoal.", step)
            stop_reason = "no_response"
            total_steps = step
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            logger.warning("Empty action at step %d, ending subgoal.", step)
            stop_reason = "empty_action"
            total_steps = step
            break

        if any(o != 0.0 for o in openvla_pose_origin):
            yaw_origin_rad = math.radians(openvla_pose_origin[3])
            reframed_poses = []
            for pose in action_poses:
                if isinstance(pose, (list, tuple)) and len(pose) >= 4:
                    reframed_poses.append([
                        float(pose[0]) + openvla_pose_origin[0],
                        float(pose[1]) + openvla_pose_origin[1],
                        float(pose[2]) + openvla_pose_origin[2],
                        float(pose[3]) + yaw_origin_rad,
                    ])
                else:
                    reframed_poses.append(pose)
            action_poses = reframed_poses

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
        converged = result is not None and result.action == "force_converge"
        steps_since_correction = step - last_correction_step
        if last_pose is not None and steps_since_correction >= check_interval:
            diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
            if all(d < SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < SMALL_DELTA_YAW:
                small_count += 1
            else:
                small_count = 0
            if small_count >= batch.ACTION_SMALL_STEPS:
                converged = True
        last_pose = list(current_pose)

        if converged:
            conv_frame = set_drone_cam_and_get_image(env, cam_id)
            if conv_frame is not None:
                conv_capture_time_unix_s = time.time()
                conv_capture_time_iso = datetime.now().isoformat(timespec="milliseconds")
                conv_path = frames_dir / f"frame_conv_{global_frame_idx:06d}.png"
                try:
                    import cv2
                    if cv2.imwrite(str(conv_path), conv_frame):
                        frame_metadata.append({
                            "frame_file": conv_path.name,
                            "frame_index": global_frame_idx,
                            "frame_type": "convergence",
                            "subgoal_index": subgoal_index,
                            "subgoal": subgoal_nl,
                            "converted_instruction": converted_instruction,
                            "subgoal_step": step,
                            "captured_at_iso": conv_capture_time_iso,
                            "captured_at_unix_s": conv_capture_time_unix_s,
                        })
                except Exception:
                    conv_path = frame_path
            else:
                conv_path = frame_path

            convergence_start = time.perf_counter()
            convergence_error: Optional[str] = None
            try:
                conv_result = monitor.on_convergence(
                    conv_path, displacement=list(current_pose),
                )
            except Exception as e:
                convergence_error = str(e)
                logger.error("monitor.on_convergence failed at step %d: %s", step, e)
                conv_result = DiaryCheckResult(
                    action="stop", new_instruction="",
                    reasoning="LLM error on convergence",
                    diary_entry="",
                    completion_pct=monitor.last_completion_pct,
                )
            convergence_end = time.perf_counter()
            telemetry_log.append({
                "event_type": "vlm_inference",
                "query_type": "diary_on_convergence",
                "subgoal_index": subgoal_index,
                "subgoal": subgoal_nl,
                "subgoal_step": step,
                "model": monitor_model,
                "duration_ms": (convergence_end - convergence_start) * 1000.0,
                "success": convergence_error is None,
                "error": convergence_error,
                "result_action": conv_result.action if conv_result is not None else None,
                "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                "timestamp_unix_s": time.time(),
            })

            if conv_result.action == "stop":
                logger.info(
                    "Subgoal complete on convergence at step %d: %s",
                    step, conv_result.reasoning,
                )
                in_correction = False
                stop_reason = "monitor_complete"
                break

            if conv_result.new_instruction:
                logger.info(
                    "Supervisor %s at step %d: '%s'",
                    conv_result.action, step, conv_result.new_instruction,
                )
                override_history.append({
                    "step": step,
                    "type": f"convergence_{conv_result.action}",
                    "old_instruction": current_instruction,
                    "new_instruction": conv_result.new_instruction,
                    "reasoning": conv_result.reasoning,
                })
                current_instruction = conv_result.new_instruction
                in_correction = True
                last_correction_step = step
                openvla_pose_origin = list(current_pose)
                small_count = 0
                last_pose = None
                batch.reset_model(server_url)
            else:
                logger.info("Convergence at step %d (no corrective command).", step)
                in_correction = False
                stop_reason = "convergence_no_command"
                break

        step += 1
    else:
        stop_reason = "max_steps"
        total_steps = max_steps

    # Save diary summary for this subgoal
    diary_summary = {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "diary": monitor.diary,
        "override_history": override_history,
        "corrections_used": monitor.corrections_used,
        "last_completion_pct": monitor.last_completion_pct,
        "high_water_mark": monitor.high_water_mark,
        "parse_failures": monitor.parse_failures,
        "vlm_calls": monitor.vlm_calls,
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "in_correction_at_end": in_correction,
    }
    with open(subgoal_dir / "diary_summary.json", "w") as f:
        json.dump(diary_summary, f, indent=2)

    monitor.cleanup()

    # Compute next subgoal's world-space origin from current pose
    next_origin_x, next_origin_y, next_origin_z, next_origin_yaw = relative_pose_to_world(
        origin_x, origin_y, origin_z, origin_yaw, current_pose,
    )

    return {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "total_steps": total_steps,
        "stop_reason": stop_reason,
        "corrections_used": monitor.corrections_used,
        "last_completion_pct": monitor.last_completion_pct,
        "high_water_mark": monitor.high_water_mark,
        "vlm_calls": monitor.vlm_calls,
        "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
    }


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
) -> Dict[str, Any]:
    """Plan with LTL, then execute each subgoal with diary monitoring.

    Returns the run_info dict written to disk.
    """
    from modules.llm_user_interface import LLM_User_Interface
    from modules.ltl_planner import LTL_Symbolic_Planner

    instruction = task["instruction"]
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps_per_subgoal = task["max_steps_per_subgoal"]
    check_interval = task["diary_check_interval"]
    max_corrections = task["max_corrections"]

    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # --- Teleport drone to initial position ---
    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3],
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[4] - 180,
    )
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    start_ts = datetime.now().isoformat()

    # --- LTL planning phase ---
    logger.info("Planning instruction: '%s'", instruction)
    llm_interface = LLM_User_Interface(model=llm_model)
    planner = LTL_Symbolic_Planner(llm_interface)
    planner.plan_from_natural_language(instruction)

    ltl_plan = {
        "ltl_nl_formula": llm_interface.ltl_nl_formula.get("ltl_nl_formula", ""),
        "pi_predicates": dict(planner.pi_map),
    }
    logger.info("LTL plan: %s", json.dumps(ltl_plan, indent=2))

    # --- Subgoal execution loop ---
    subgoal_summaries: List[Dict[str, Any]] = []
    trajectory_log: List[Dict[str, Any]] = []
    frame_metadata: List[Dict[str, Any]] = []
    telemetry_log: List[Dict[str, Any]] = []
    total_frame_count = 0
    subgoal_index = 0

    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[4]

    current_subgoal = planner.get_next_predicate()
    if current_subgoal is None:
        raise RuntimeError("LTL planning produced no subgoals.")

    while current_subgoal is not None:
        subgoal_index += 1
        safe_name = _sanitize_name(current_subgoal)
        subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"

        logger.info(
            "--- Subgoal %d: '%s' ---", subgoal_index, current_subgoal,
        )

        subgoal_result = _run_subgoal(
            env=env,
            batch=batch,
            server_url=server_url,
            subgoal_nl=current_subgoal,
            monitor_model=monitor_model,
            check_interval=check_interval,
            max_steps=max_steps_per_subgoal,
            max_corrections=max_corrections,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            origin_yaw=origin_yaw,
            drone_cam_id=drone_cam_id,
            frames_dir=frames_dir,
            subgoal_dir=subgoal_dir,
            subgoal_index=subgoal_index,
            frame_offset=total_frame_count,
            trajectory_log=trajectory_log,
            frame_metadata=frame_metadata,
            telemetry_log=telemetry_log,
        )

        total_frame_count += subgoal_result["total_steps"]
        subgoal_summaries.append(subgoal_result)

        logger.info(
            "Subgoal %d finished: stop_reason=%s, steps=%d, completion=%.2f",
            subgoal_index,
            subgoal_result["stop_reason"],
            subgoal_result["total_steps"],
            subgoal_result.get("last_completion_pct", 0.0),
        )

        # Advance planner state
        planner.advance_state(current_subgoal)

        # Update world-space origin for the next subgoal
        next_origin = subgoal_result["next_origin"]
        origin_x, origin_y, origin_z, origin_yaw = (
            next_origin[0], next_origin[1], next_origin[2], next_origin[3],
        )

        current_subgoal = planner.get_next_predicate()
        if current_subgoal is not None:
            logger.info("Advancing to next subgoal: '%s'", current_subgoal)

    logger.info("All subgoals processed (%d total).", subgoal_index)

    end_ts = datetime.now().isoformat()

    # --- Save trajectory log ---
    with open(run_dir / "trajectory_log.json", "w") as f:
        json.dump(trajectory_log, f, indent=2)

    # --- Save frame to subgoal mapping ---
    with open(run_dir / "frame_metadata.json", "w") as f:
        json.dump(frame_metadata, f, indent=2)

    # --- Save per-query inference telemetry ---
    with open(run_dir / "inference_telemetry.json", "w") as f:
        json.dump(telemetry_log, f, indent=2)

    # --- Optional playback mp4 ---
    playback_mp4: Optional[Path] = None
    if save_mp4:
        try:
            import playback_fpv
            playback_mp4 = playback_fpv.save_run_directory_mp4(run_dir, fps=mp4_fps)
        except Exception as e:
            logger.warning("Could not write playback.mp4: %s", e)

    # --- Save run info ---
    total_steps_all = sum(s["total_steps"] for s in subgoal_summaries)
    total_vlm_calls = sum(s.get("vlm_calls", 0) for s in subgoal_summaries)
    total_corrections = sum(s.get("corrections_used", 0) for s in subgoal_summaries)

    run_info: Dict[str, Any] = {
        "task": task,
        "llm_model": llm_model,
        "monitor_model": monitor_model,
        "models": {
            "ltl_nl_planning": llm_model,
            "subgoal_converter": monitor_model,
            "live_diary_monitor": monitor_model,
            "openvla_predict_url": server_url,
        },
        "ltl_plan": ltl_plan,
        "subgoal_count": subgoal_index,
        "subgoal_summaries": subgoal_summaries,
        "total_steps": total_steps_all,
        "total_vlm_calls": total_vlm_calls,
        "total_corrections": total_corrections,
        "playback_mp4": str(playback_mp4) if playback_mp4 else None,
        "start_time": start_ts,
        "end_time": end_ts,
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    return run_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Integrated LTL planner + diary monitor control loop",
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
        help="Run single task from tasks/system_tasks/",
    )
    mode.add_argument(
        "--run_all_tasks",
        action="store_true",
        help="Run all JSON tasks in tasks/system_tasks/",
    )

    parser.add_argument(
        "--initial-position",
        type=str,
        default=DEFAULT_INITIAL_POSITION,
        metavar="x,y,z,yaw",
        help="Initial position (for -c mode). Default: %(default)s",
    )

    # Sim / server
    parser.add_argument("-e", "--env_id", default=DOWNTOWN_ENV_ID, help="Environment ID")
    parser.add_argument("-t", "--time_dilation", type=int, default=DEFAULT_TIME_DILATION)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-p", "--server_port", type=int, default=DEFAULT_SERVER_PORT)

    # LLM / monitor models
    parser.add_argument(
        "--llm_model",
        default="gpt-4o-mini",
        help="LLM for LTL decomposition (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--monitor_model",
        default="gpt-5.4",
        help="VLM for diary monitor and subgoal converter (default: gpt-5.4)",
    )

    # Diary parameters (override task JSON values)
    parser.add_argument(
        "--diary-check-interval",
        type=int,
        default=None,
        help=f"Steps between diary checkpoints (default from task JSON or {DEFAULT_DIARY_CHECK_INTERVAL})",
    )
    parser.add_argument(
        "--max-steps-per-subgoal",
        type=int,
        default=None,
        help=f"Max steps per subgoal (default from task JSON or {DEFAULT_MAX_STEPS_PER_SUBGOAL})",
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
        default=str(INTEGRATION_RESULTS_DIR),
        help="Base directory for results (default: results/integration_results)",
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
        "--use-default-cam",
        action="store_true",
        help=f"Use default drone camera (ID {DRONE_CAM_ID}), skip interactive selection",
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

    tasks = _resolve_tasks(args)

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
        initial_pos_for_cam = (
            tasks[0]["initial_pos"] if tasks
            else normalize_initial_pos(parse_position(DEFAULT_INITIAL_POSITION))
        )
        drone_cam_id = interactive_camera_select(env, initial_pos_for_cam, batch)

    for idx, task in enumerate(tasks):
        logger.info(
            "\n===== Task %d/%d: '%s' =====",
            idx + 1, len(tasks), task["instruction"][:80],
        )
        start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_name = (
            f"run_{start_time}"
            if len(tasks) == 1
            else f"run_{start_time}_{idx:02d}"
        )
        try:
            run_dir = results_base / run_name
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

    env.close()


if __name__ == "__main__":
    main()
