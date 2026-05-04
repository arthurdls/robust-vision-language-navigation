#!/usr/bin/env python3
"""
Condition 4: Single-Frame Monitor (No Temporal Context).

Uses the full LTL planner and SubgoalConverter. At each checkpoint interval,
sends ONLY the current camera frame and the subgoal text to the VLM. No diary
history, no image grid, no displacement data, no completion percentage tracking.

On convergence, uses the same single-frame query to decide complete vs.
stopped_short and issue corrections.

OpenVLA server must be running: python scripts/start_server.py

Usage (from repo root):
  python scripts/run_condition4_single_frame.py --task third_task.json
  python scripts/run_condition4_single_frame.py --run_all_tasks
  python scripts/run_condition4_single_frame.py -c "Go to the tree then land" --initial-position -181,7331,876,-89
"""

import argparse
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
from rvln.eval.task_utils import (
    get_completed_task_ids,
    resolve_eval_tasks,
    sanitize_run_label,
)
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
    relative_pose_to_world,
    set_drone_cam_and_get_image,
    setup_sim_env,
    state_for_openvla,
)

SHARED_TASKS_DIR = REPO_ROOT / "tasks"
CONDITION4_RESULTS_DIR = REPO_ROOT / "results" / "condition4"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-frame VLM prompts (no temporal context)
# ---------------------------------------------------------------------------

SINGLE_FRAME_SYSTEM_PROMPT = """\
You are a completion monitor for an autonomous drone executing a single subgoal.
You see a single frame from the drone's first-person camera (no history).
Assess whether the subgoal is complete based on this one frame alone."""

SINGLE_FRAME_CHECK_PROMPT = """\
Subgoal: {subgoal}
{constraints_block}
Look at this single frame from the drone's first-person camera.
Is the drone currently at the goal described by the subgoal?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "should_stop": true/false,
  "constraint_violated": true/false
}}

- "complete": true ONLY if the subgoal has been fully accomplished based on
  this single frame.
- "completion_percentage": your best estimate of progress (0.0 to 1.0).
  NEVER set 1.0 unless highly confident. Cap at 0.95 when unsure.
- "should_stop": true if the drone appears off-track or heading toward a
  collision. The drone will be stopped and a corrective instruction issued.
- "constraint_violated": true if any active constraint listed above appears
  violated in this frame. false if no constraints are listed or none violated."""

SINGLE_FRAME_CONVERGENCE_PROMPT = """\
Subgoal: {subgoal}
{constraints_block}
The drone has stopped moving. Look at this single frame from the drone's camera.
Is the subgoal complete? If not, what single corrective command should be issued?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "diagnosis": "stopped_short" or "overshot" or "complete" or "constraint_violated",
  "corrective_instruction": "..." or null,
  "constraint_violated": true/false
}}

- "complete": true ONLY if you are highly confident the subgoal is done.
- "diagnosis": "complete" if done, "stopped_short" if the drone needs to
  keep going, "overshot" if the drone went past the goal.
- "corrective_instruction": REQUIRED if not complete. A single-action drone
  command to fix the biggest gap. null only if complete.
- "constraint_violated": true if any active constraint appears violated."""


# ---------------------------------------------------------------------------
# Single-frame monitor helper
# ---------------------------------------------------------------------------

def _parse_json_response(response: str) -> Optional[dict]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _format_constraints_block(constraints) -> str:
    if not constraints:
        return ""
    lines = ["Active constraints (must be maintained throughout):"]
    for c in constraints:
        if hasattr(c, "polarity"):
            label = "AVOID" if c.polarity == "negative" else "MAINTAIN"
            lines.append(f"  - {label}: {c.description}")
        else:
            lines.append(f"  - {c}")
    lines.append("")
    return "\n".join(lines)


def _query_single_frame(frame_path: Path, prompt: str, llm, label: str = "single_frame_check"):
    """Send a single frame + prompt to the VLM. Returns (response_text, call_record)."""
    import time as _time
    from rvln.ai.utils.vision import query_vlm, build_frame_grid
    grid = build_frame_grid([frame_path])
    t0 = _time.time()
    response = query_vlm(grid, prompt, llm=llm, system_prompt=SINGLE_FRAME_SYSTEM_PROMPT)
    rtt = _time.time() - t0
    usage = llm.last_usage
    call_record = {
        "label": label,
        "rtt_s": round(rtt, 3),
        "model": usage.get("model", ""),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    return response, call_record


# ---------------------------------------------------------------------------
# Per-subgoal control loop with single-frame VLM checks
# ---------------------------------------------------------------------------

def _run_subgoal(
    env, batch, server_url, subgoal_nl, monitor_model, llm_model, check_interval,
    max_steps, max_corrections, origin_x, origin_y, origin_z, origin_yaw,
    drone_cam_id, frames_dir, subgoal_dir, frame_offset, trajectory_log,
    constraints=None,
):
    from rvln.ai.subgoal_converter import SubgoalConverter
    from rvln.ai.utils.llm_providers import LLMFactory

    subgoal_dir.mkdir(parents=True, exist_ok=True)

    converter = SubgoalConverter(model=llm_model)
    conversion = converter.convert(subgoal_nl)
    converted_instruction = conversion.instruction
    current_instruction = converted_instruction

    if monitor_model.startswith("gemini"):
        llm = LLMFactory.create("gemini", model=monitor_model)
    else:
        llm = LLMFactory.create("openai", model=monitor_model)

    current_pose = [0.0, 0.0, 0.0, 0.0]
    openvla_pose_origin = [0.0, 0.0, 0.0, 0.0]
    last_pose = None
    small_count = 0
    constraints_block = _format_constraints_block(constraints)
    override_history = []
    corrections_used = 0
    vlm_calls = 0
    vlm_call_records = list(converter.llm_call_records)
    last_correction_step = -check_interval
    stop_reason = "max_steps"
    total_steps = 0
    constraint_violation_count = 0

    batch.reset_model(server_url)
    step = 0

    while step < max_steps:
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

        # Single-frame VLM check at checkpoint intervals
        if step > 0 and step % check_interval == 0:
            prompt = SINGLE_FRAME_CHECK_PROMPT.format(
                subgoal=subgoal_nl, constraints_block=constraints_block,
            )
            try:
                response_text, call_rec = _query_single_frame(
                    frame_path, prompt, llm, label="single_frame_check",
                )
                call_rec["step"] = step
                vlm_call_records.append(call_rec)
                vlm_calls += 1
                parsed = _parse_json_response(response_text)
                if parsed is not None:
                    if parsed.get("constraint_violated", False):
                        constraint_violation_count += 1
                        override_history.append({
                            "step": step,
                            "type": "constraint_violation",
                            "constraint_violated": True,
                        })
                    if parsed.get("complete", False):
                        logger.info("Single-frame check: complete at step %d", step)
                        stop_reason = "monitor_complete"
                        total_steps = step
                        break
                    if parsed.get("should_stop", False):
                        logger.info("Single-frame check: should_stop at step %d", step)
                        # Force convergence to trigger correction
            except Exception as e:
                logger.error("Single-frame check failed at step %d: %s", step, e)

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

        # Convergence detection
        converged = False
        steps_since_correction = step - last_correction_step
        if last_pose is not None and steps_since_correction >= check_interval:
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

            if corrections_used >= max_corrections:
                logger.warning("Max corrections (%d) exhausted.", max_corrections)
                stop_reason = "max_corrections"
                break

            prompt = SINGLE_FRAME_CONVERGENCE_PROMPT.format(
                subgoal=subgoal_nl, constraints_block=constraints_block,
            )
            try:
                response_text, call_rec = _query_single_frame(
                    conv_path, prompt, llm, label="single_frame_convergence",
                )
                call_rec["step"] = step
                vlm_call_records.append(call_rec)
                vlm_calls += 1
                parsed = _parse_json_response(response_text)
            except Exception as e:
                logger.error("Single-frame convergence failed: %s", e)
                stop_reason = "convergence_error"
                break

            if parsed is None:
                logger.warning("Unparseable convergence response, stopping.")
                stop_reason = "parse_error"
                break

            if parsed.get("constraint_violated", False):
                constraint_violation_count += 1
                override_history.append({
                    "step": step,
                    "type": "constraint_violation",
                    "constraint_violated": True,
                })

            if parsed.get("complete", False) or parsed.get("diagnosis") == "complete":
                logger.info("Single-frame convergence: complete at step %d", step)
                stop_reason = "monitor_complete"
                break

            corrective = parsed.get("corrective_instruction") or ""
            if corrective:
                corrections_used += 1
                override_history.append({
                    "step": step,
                    "type": f"convergence_{parsed.get('diagnosis', 'unknown')}",
                    "old_instruction": current_instruction,
                    "new_instruction": corrective,
                })
                current_instruction = corrective
                last_correction_step = step
                openvla_pose_origin = list(current_pose)
                small_count = 0
                last_pose = None
                batch.reset_model(server_url)
            else:
                logger.info("Convergence at step %d (no corrective command).", step)
                stop_reason = "convergence_no_command"
                break

        step += 1
    else:
        stop_reason = "max_steps"
        total_steps = max_steps

    subgoal_summary = {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "override_history": override_history,
        "corrections_used": corrections_used,
        "vlm_call_count": vlm_calls,
        "vlm_call_records": vlm_call_records,
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "constraint_violation_count": constraint_violation_count,
    }
    with open(subgoal_dir / "subgoal_summary.json", "w") as f:
        json.dump(subgoal_summary, f, indent=2)

    next_origin_x, next_origin_y, next_origin_z, next_origin_yaw = relative_pose_to_world(
        origin_x, origin_y, origin_z, origin_yaw, current_pose,
    )

    return {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "total_steps": total_steps,
        "stop_reason": stop_reason,
        "corrections_used": corrections_used,
        "vlm_call_count": vlm_calls,
        "vlm_call_records": vlm_call_records,
        "constraint_violation_count": constraint_violation_count,
        "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
    }


# ---------------------------------------------------------------------------
# Integrated control loop
# ---------------------------------------------------------------------------

def run_single_frame_control_loop(
    env, batch, task, server_url, run_dir,
    llm_model, monitor_model, drone_cam_id,
    save_mp4=False, mp4_fps=10.0,
    seed=DEFAULT_SEED, time_dilation=DEFAULT_TIME_DILATION, env_id="",
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

    env.teleport(initial_pos[0:3], initial_pos[3])
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
    origin_yaw = initial_pos[3]

    current_subgoal = planner.get_next_predicate()
    if current_subgoal is None:
        raise RuntimeError("LTL planning produced no subgoals.")

    while current_subgoal is not None:
        subgoal_index += 1
        safe_name = sanitize_run_label(current_subgoal, fallback="subgoal")
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
        "completed": True,
        "condition": "condition4_single_frame",
        "task": task,
        "seed": seed,
        "time_dilation": time_dilation,
        "env_id": env_id,
        "server_url": server_url,
        "drone_cam_id": drone_cam_id,
        "llm_model": llm_model,
        "monitor_model": monitor_model,
        "models": {
            "ltl_nl_planning": llm_model,
            "subgoal_converter": llm_model,
            "single_frame_vlm": monitor_model,
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
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Condition 4: Single-frame VLM monitor (no temporal context)",
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
    parser.add_argument("--diary-check-interval", type=int, default=None)
    parser.add_argument("--max-steps-per-subgoal", type=int, default=None)
    parser.add_argument("--max-corrections", type=int, default=None)
    parser.add_argument("-o", "--results_dir", default=str(CONDITION4_RESULTS_DIR))
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

    env = setup_sim_env(int(args.time_dilation), int(args.seed), batch,
                        sim_host=args.sim_host, sim_api_port=args.sim_api_port)
    map_info = env.get_map_info()
    results_base = Path(args.results_dir) / map_info.task_dir_name
    results_base.mkdir(parents=True, exist_ok=True)
    tasks = resolve_eval_tasks(args, map_info, SHARED_TASKS_DIR, overrides={"max_steps_per_subgoal": "max_steps_per_subgoal", "diary_check_interval": "diary_check_interval", "max_corrections": "max_corrections"})

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
            task_label = task.get("task_id") or sanitize_run_label(task["instruction"], max_len=30, fallback="subgoal")
            run_name = f"c4_single_frame__{task_label}__{ts}"
            try:
                run_dir = results_base / run_name
                run_info = run_single_frame_control_loop(
                    env=env, batch=batch, task=task, server_url=server_url,
                    run_dir=run_dir, llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    drone_cam_id=drone_cam_id, save_mp4=args.save_mp4, mp4_fps=args.mp4_fps,
                    seed=args.seed, time_dilation=args.time_dilation,
                    env_id=map_info.env_id,
                )
                logger.info("Run saved to %s (%d subgoals, %d total steps)",
                            run_dir, run_info["subgoal_count"], run_info["total_steps"])
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
