#!/usr/bin/env python3
"""
Condition 5: Image-Grid Only Monitor (No Text Diary).

Uses the full LTL planner and SubgoalConverter. The GoalAdherenceMonitor runs
normally for local checkpoints (2-frame VLM queries generate diary entries),
but the GLOBAL and CONVERGENCE prompts are modified to remove the text diary
and displacement data. The VLM sees only the subgoal text and the 3x3 image
grid for progress assessment.

This tests whether VLMs can infer progress/completion from visual history
alone without explicit textual state tracking.

OpenVLA server must be running: python scripts/start_server.py

Usage (from repo root):
  python scripts/run_condition5_grid_only.py --task third_task.json
  python scripts/run_condition5_grid_only.py --run_all_tasks
  python scripts/run_condition5_grid_only.py -c "Go to the tree then land" --initial-position -181,7331,876,-89
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
from rvln.paths import (
    BATCH_SCRIPT,
    REPO_ROOT,
    UAV_FLOW_EVAL,
)
from rvln.maps import validate_task_map
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

CONDITION5_TASKS_DIR = REPO_ROOT / "tasks" / "condition5"
CONDITION5_RESULTS_DIR = REPO_ROOT / "results" / "condition5"

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


# ---------------------------------------------------------------------------
# Modified prompts: grid only, no diary text, no displacement
# ---------------------------------------------------------------------------

GRID_ONLY_GLOBAL_PROMPT = """\
Subgoal: {subgoal}
{constraints_block}
Previous estimated completion: {prev_completion_pct}

The grid shows up to the 9 most recent sampled frames (left to right, top to
bottom, in temporal order).

Based on the visual progression in the grid of sampled frames, respond with
EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "on_track": true/false,
  "should_stop": true/false,
  "constraint_violated": true/false
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished. Do NOT mark complete for partial progress.
- "completion_percentage": your best estimate of how close the subgoal is to
  completion (0.0 = not started, 1.0 = fully done). NEVER set 1.0 unless you
  are highly confident. Cap at 0.95 when unsure.
- "on_track": true if the drone is making any progress toward the subgoal.
- "should_stop": true only if the drone is actively making things worse.
- "constraint_violated": true if any active constraint listed above has been
  violated. false if no constraints are listed or none have been violated."""

GRID_ONLY_CONVERGENCE_PROMPT = """\
Subgoal: {subgoal}
{constraints_block}
Previous estimated completion: {prev_completion_pct}

The drone has stopped moving. The grid shows up to the 9 most recent sampled
frames (left to right, top to bottom, in temporal order).

Given the visual progression in the sampled frames, is the subgoal complete?
If not, did the drone stop short or overshoot?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "diagnosis": "stopped_short" or "overshot" or "complete" or "constraint_violated",
  "corrective_instruction": "..." or null,
  "constraint_violated": true/false
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished.
- "diagnosis": "complete" if done, "stopped_short" if the drone needs to keep
  going, "overshot" if the drone went past the goal.
- "corrective_instruction": REQUIRED if not complete. A single-action drone
  command. null only if complete.
- "constraint_violated": true if any active constraint has been violated."""


# ---------------------------------------------------------------------------
# Monkey-patch the goal adherence monitor prompts for this condition
# ---------------------------------------------------------------------------

def _patch_prompts():
    """Replace global and convergence prompt templates in goal_adherence_monitor module."""
    import rvln.ai.goal_adherence_monitor as gam_module
    gam_module.GLOBAL_PROMPT_TEMPLATE = GRID_ONLY_GLOBAL_PROMPT
    gam_module.CONVERGENCE_PROMPT_TEMPLATE = GRID_ONLY_CONVERGENCE_PROMPT


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

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
        "diary_check_interval": int(data.get("diary_check_interval", DEFAULT_DIARY_CHECK_INTERVAL)),
        "max_corrections": int(data.get("max_corrections", DEFAULT_MAX_CORRECTIONS)),
    }
    for passthrough_key in ("task_id", "category", "difficulty", "region", "notes"):
        if passthrough_key in data:
            result[passthrough_key] = data[passthrough_key]
    return result


def _resolve_tasks(args: argparse.Namespace, map_info) -> List[Dict[str, Any]]:
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
        initial_pos_str = getattr(args, "initial_position", None) or map_info.default_position
        return [{
            "instruction": cmd.strip(),
            "initial_pos": parse_position(initial_pos_str),
            "max_steps_per_subgoal": args.max_steps_per_subgoal,
            "diary_check_interval": args.diary_check_interval,
            "max_corrections": args.max_corrections,
        }]

    tasks_dir = CONDITION5_TASKS_DIR / map_info.task_dir_name

    if task_file is not None:
        validate_task_map(task_file, map_info)
        path = Path(task_file)
        if not path.is_absolute():
            if len(path.parts) > 1:
                path = CONDITION5_TASKS_DIR / path
            else:
                path = tasks_dir / path.name
        if not path.exists():
            raise SystemExit(f"Task file not found: {path}")
        task = _load_task(path)
        task["max_steps_per_subgoal"] = args.max_steps_per_subgoal or task["max_steps_per_subgoal"]
        task["diary_check_interval"] = args.diary_check_interval or task["diary_check_interval"]
        task["max_corrections"] = args.max_corrections or task["max_corrections"]
        return [task]

    tasks_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(tasks_dir / "*.json")))
    if not json_files:
        raise SystemExit(f"No JSON files found in {tasks_dir}")
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
    clean = text.lower().replace(" ", "_")
    safe = "".join(c for c in clean if c.isalnum() or c == "_")
    return safe[:max_len] or "subgoal"


# ---------------------------------------------------------------------------
# Per-subgoal control loop (reuses run_integration._run_subgoal logic with
# patched prompts and no constraints)
# ---------------------------------------------------------------------------

def _run_subgoal(
    env, batch, server_url, subgoal_nl, monitor_model, llm_model, check_interval,
    max_steps, max_corrections, origin_x, origin_y, origin_z, origin_yaw,
    drone_cam_id, frames_dir, subgoal_dir, frame_offset, trajectory_log,
    check_interval_s=None, max_seconds=None, constraints=None,
):
    from rvln.ai.goal_adherence_monitor import DiaryCheckResult, GoalAdherenceMonitor
    from rvln.ai.subgoal_converter import SubgoalConverter

    use_async = check_interval_s is not None
    subgoal_dir.mkdir(parents=True, exist_ok=True)
    diary_artifacts = subgoal_dir / "diary_artifacts"
    diary_artifacts.mkdir(parents=True, exist_ok=True)

    converter = SubgoalConverter(model=llm_model)
    conversion = converter.convert(subgoal_nl)
    converted_instruction = conversion.instruction
    current_instruction = converted_instruction

    monitor = GoalAdherenceMonitor(
        subgoal=subgoal_nl,
        check_interval=check_interval,
        model=monitor_model,
        artifacts_dir=diary_artifacts,
        max_corrections=max_corrections,
        check_interval_s=check_interval_s,
        constraints=constraints,
    )

    current_pose = [0.0, 0.0, 0.0, 0.0]
    openvla_pose_origin = [0.0, 0.0, 0.0, 0.0]
    last_pose = None
    small_count = 0
    override_history = []
    in_correction = False
    last_correction_step = -check_interval
    last_correction_time = time.time() if use_async else None
    subgoal_start_time = time.time()
    stop_reason = "max_steps"
    total_steps = 0

    batch.reset_model(server_url)
    result = None
    step = 0

    while step < max_steps:
        if max_seconds is not None and (time.time() - subgoal_start_time) >= max_seconds:
            stop_reason = "max_seconds"
            total_steps = step
            break

        if use_async:
            async_result = monitor.poll_result()
            if async_result is not None:
                if async_result.action == "stop":
                    stop_reason = "monitor_complete"
                    total_steps = step
                    break
                if async_result.action == "ask_help":
                    stop_reason = "ask_help"
                    total_steps = step
                    break
                if async_result.action == "force_converge" and async_result.constraint_violated:
                    override_history.append({
                        "step": step,
                        "type": "constraint_violation",
                        "reasoning": async_result.reasoning,
                        "constraint_violated": True,
                    })

        image = set_drone_cam_and_get_image(env, drone_cam_id)
        if image is None:
            stop_reason = "no_image"
            total_steps = step
            break

        global_frame_idx = frame_offset + step
        frame_path = frames_dir / f"frame_{global_frame_idx:06d}.png"
        try:
            import cv2
            cv2.imwrite(str(frame_path), image)
        except Exception:
            pass

        try:
            result = monitor.on_frame(frame_path, displacement=list(current_pose))
        except Exception as e:
            logger.error("monitor.on_frame failed at step %d: %s", step, e)
            result = DiaryCheckResult(
                action="continue", new_instruction="", reasoning="",
                diary_entry="", completion_pct=monitor.last_completion_pct,
            )

        if not use_async:
            if result.action == "stop":
                stop_reason = "monitor_complete"
                total_steps = step
                break
            if result.action == "ask_help":
                stop_reason = "ask_help"
                total_steps = step
                break

        openvla_pose = [c - o for c, o in zip(current_pose, openvla_pose_origin)]
        response = batch.send_prediction_request(
            image=Image.fromarray(image),
            proprio=state_for_openvla(openvla_pose),
            instr=current_instruction.strip().lower(),
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
                env, action_poses, origin_x, origin_y, origin_z, origin_yaw,
                trajectory_log=trajectory_log, sleep_s=0.1, drone_cam_id=drone_cam_id,
            )
        except Exception as e:
            logger.error("Error executing action at step %d: %s", step, e)
            stop_reason = "action_error"
            total_steps = step
            break

        total_steps = step + 1

        converged = False
        if not use_async:
            if result is not None and result.action == "force_converge":
                converged = True
                if result.constraint_violated:
                    override_history.append({
                        "step": step,
                        "type": "constraint_violation",
                        "reasoning": result.reasoning,
                        "constraint_violated": True,
                    })
            steps_since_correction = step - last_correction_step
            if last_pose is not None and steps_since_correction >= check_interval:
                diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
                if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                    small_count += 1
                else:
                    small_count = 0
                if small_count >= batch.ACTION_SMALL_STEPS:
                    converged = True
        else:
            elapsed_since_correction = time.time() - last_correction_time
            if last_pose is not None and elapsed_since_correction >= check_interval_s:
                diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
                if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                    small_count += 1
                else:
                    small_count = 0
                if small_count >= batch.ACTION_SMALL_STEPS:
                    converged = True
        last_pose = list(current_pose)

        if converged:
            conv_frame = set_drone_cam_and_get_image(env, drone_cam_id)
            conv_path = frame_path
            if conv_frame is not None:
                conv_path = frames_dir / f"frame_conv_{global_frame_idx:06d}.png"
                try:
                    import cv2
                    cv2.imwrite(str(conv_path), conv_frame)
                except Exception:
                    conv_path = frame_path

            try:
                conv_result = monitor.on_convergence(conv_path, displacement=list(current_pose))
            except Exception as e:
                logger.error("monitor.on_convergence failed: %s", e)
                conv_result = DiaryCheckResult(
                    action="stop", new_instruction="",
                    reasoning="LLM error on convergence", diary_entry="",
                    completion_pct=monitor.last_completion_pct,
                )

            if conv_result.action == "stop":
                in_correction = False
                stop_reason = "monitor_complete"
                break
            if conv_result.action == "ask_help":
                stop_reason = "ask_help"
                total_steps = step
                break

            if conv_result.new_instruction:
                override_history.append({
                    "step": step,
                    "type": f"convergence_{conv_result.action}",
                    "old_instruction": current_instruction,
                    "new_instruction": conv_result.new_instruction,
                })
                current_instruction = conv_result.new_instruction
                in_correction = True
                last_correction_step = step
                if use_async:
                    last_correction_time = time.time()
                openvla_pose_origin = list(current_pose)
                small_count = 0
                last_pose = None
                batch.reset_model(server_url)
            else:
                in_correction = False
                stop_reason = "convergence_no_command"
                break

        step += 1
    else:
        stop_reason = "max_steps"
        total_steps = max_steps

    all_vlm_call_records = list(monitor.vlm_rtts) + list(converter.llm_call_records)
    constraint_violation_count = sum(
        1 for o in override_history
        if o.get("constraint_violated", False)
    )

    diary_summary = {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "diary": monitor.diary,
        "override_history": override_history,
        "corrections_used": monitor.corrections_used,
        "last_completion_pct": monitor.last_completion_pct,
        "peak_completion": monitor.peak_completion,
        "parse_failures": monitor.parse_failures,
        "vlm_call_count": monitor.vlm_calls,
        "vlm_call_records": list(monitor.vlm_rtts),
        "converter_call_records": list(converter.llm_call_records),
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "constraint_violation_count": constraint_violation_count,
    }
    with open(subgoal_dir / "diary_summary.json", "w") as f:
        json.dump(diary_summary, f, indent=2)

    monitor.cleanup()

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
        "peak_completion": monitor.peak_completion,
        "vlm_call_count": monitor.vlm_calls,
        "vlm_call_records": all_vlm_call_records,
        "parse_failures": monitor.parse_failures,
        "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
        "constraints": [{"description": c.description, "polarity": c.polarity} for c in (constraints or [])],
        "constraint_violation_count": constraint_violation_count,
    }


# ---------------------------------------------------------------------------
# Integrated control loop
# ---------------------------------------------------------------------------

def run_grid_only_control_loop(
    env, batch, task, server_url, run_dir,
    llm_model, monitor_model, drone_cam_id,
    save_mp4=False, mp4_fps=10.0,
    check_interval_s=None, max_seconds=None,
    seed=DEFAULT_SEED, time_dilation=DEFAULT_TIME_DILATION,
    env_id="", diary_mode=DEFAULT_DIARY_MODE,
):
    from rvln.ai.llm_interface import LLMUserInterface
    from rvln.ai.ltl_planner import LTLSymbolicPlanner

    instruction = task["instruction"]
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps_per_subgoal = task["max_steps_per_subgoal"]
    check_interval = task["diary_check_interval"]
    max_corrections = task["max_corrections"]

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

    while current_subgoal is not None:
        subgoal_index += 1
        safe_name = _sanitize_name(current_subgoal)
        subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"

        active_constraints = planner.get_active_constraints()

        logger.info("--- Subgoal %d: '%s' ---", subgoal_index, current_subgoal)

        subgoal_result = _run_subgoal(
            env=env, batch=batch, server_url=server_url,
            subgoal_nl=current_subgoal, monitor_model=monitor_model, llm_model=llm_model,
            check_interval=check_interval, max_steps=max_steps_per_subgoal,
            max_corrections=max_corrections,
            origin_x=origin_x, origin_y=origin_y,
            origin_z=origin_z, origin_yaw=origin_yaw,
            drone_cam_id=drone_cam_id, frames_dir=frames_dir,
            subgoal_dir=subgoal_dir, frame_offset=total_frame_count,
            trajectory_log=trajectory_log,
            check_interval_s=check_interval_s, max_seconds=max_seconds,
            constraints=active_constraints,
        )

        total_frame_count += subgoal_result["total_steps"]
        subgoal_summaries.append(subgoal_result)

        planner.advance_state(current_subgoal)

        next_origin = subgoal_result["next_origin"]
        origin_x, origin_y, origin_z, origin_yaw = (
            next_origin[0], next_origin[1], next_origin[2], next_origin[3],
        )

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
    any_constraint_violated = any(
        s.get("constraint_violation_count", 0) > 0 for s in subgoal_summaries
    )
    end_dt = datetime.fromisoformat(end_ts)
    start_dt = datetime.fromisoformat(start_ts)
    wall_clock_seconds = (end_dt - start_dt).total_seconds()

    run_info = {
        "condition": "condition5_grid_only",
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
            "grid_only_monitor": monitor_model,
            "openvla_predict_url": server_url,
        },
        "config": {
            "max_steps_per_subgoal": max_steps_per_subgoal,
            "diary_check_interval": check_interval,
            "max_corrections": max_corrections,
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
        "total_vlm_calls": sum(s.get("vlm_call_count", 0) for s in subgoal_summaries),
        "total_corrections": sum(s.get("corrections_used", 0) for s in subgoal_summaries),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "vlm_call_records": all_vlm_records,
        "any_constraint_violated": any_constraint_violated,
        "playback_mp4": str(playback_mp4) if playback_mp4 else None,
        "start_time": start_ts,
        "end_time": end_ts,
        "wall_clock_seconds": round(wall_clock_seconds, 2),
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    return run_info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _patch_prompts()

    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Condition 5: Image-grid only monitor (no text diary, no displacement)",
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
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--monitor_model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--diary-mode", choices=("frame", "time"), default=DEFAULT_DIARY_MODE)
    parser.add_argument("--diary-check-interval", type=int, default=None)
    parser.add_argument("--diary-check-interval-s", type=float, default=DEFAULT_DIARY_CHECK_INTERVAL_S)
    parser.add_argument("--max-steps-per-subgoal", type=int, default=None)
    parser.add_argument("--max-seconds-per-subgoal", type=float, default=DEFAULT_MAX_SECONDS_PER_SUBGOAL)
    parser.add_argument("--max-corrections", type=int, default=None)
    parser.add_argument("-o", "--results_dir", default=str(CONDITION5_RESULTS_DIR))
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

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    batch = import_batch_module()
    os.chdir(str(UAV_FLOW_EVAL))

    server_url = f"http://{args.server_host}:{args.server_port}/predict"
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    env = setup_sim_env(int(args.time_dilation), int(args.seed), batch,
                        sim_host=args.sim_host, sim_api_port=args.sim_api_port)
    map_info = env.get_map_info()
    tasks = _resolve_tasks(args, map_info)

    try:
        drone_cam_id = env.drone_cam_id
        if args.select_cam:
            initial_pos_for_cam = (
                tasks[0]["initial_pos"] if tasks
                else normalize_initial_pos(parse_position(map_info.default_position))
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
            run_name = f"c5_grid_only__{task_label}__{ts}"
            try:
                run_dir = results_base / run_name
                use_time_mode = args.diary_mode == "time"
                run_info = run_grid_only_control_loop(
                    env=env, batch=batch, task=task, server_url=server_url,
                    run_dir=run_dir, llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    drone_cam_id=drone_cam_id, save_mp4=args.save_mp4, mp4_fps=args.mp4_fps,
                    check_interval_s=args.diary_check_interval_s if use_time_mode else None,
                    max_seconds=args.max_seconds_per_subgoal if use_time_mode else None,
                    seed=args.seed, time_dilation=args.time_dilation,
                    env_id=map_info.env_id, diary_mode=args.diary_mode,
                )
                logger.info("Run saved to %s (%d subgoals, %d total steps)",
                            run_dir, run_info["subgoal_count"], run_info["total_steps"])
            except KeyboardInterrupt:
                logger.info("Task interrupted.")
                break
            except Exception as e:
                logger.error("Task failed: %s", e, exc_info=True)
            logger.info("===== Task %d finished =====\n", idx + 1)


if __name__ == "__main__":
    main()
