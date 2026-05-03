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
  python scripts/run_integration.py --task seq_constraint_01.json
  # All tasks in tasks/condition0/ (skips already completed)
  python scripts/run_integration.py --run_all_tasks
  # Ad-hoc command
  python scripts/run_integration.py -c "Go to the tree then land" --initial-position -181,7331,876,-89
  # Custom models and diary parameters
  python scripts/run_integration.py --task first_task.json --llm_model gpt-4o --monitor_model gpt-4o \\
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

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.config import (
    ACTION_SMALL_DELTA_POS,
    ACTION_SMALL_DELTA_YAW,
    DEFAULT_DIARY_CHECK_INTERVAL,
    DEFAULT_DIARY_CHECK_INTERVAL_S,
    DEFAULT_DIARY_MODE,
    DEFAULT_INITIAL_POSITION,
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

CONDITION0_TASKS_DIR = REPO_ROOT / "tasks" / "condition0"
CONDITION0_RESULTS_DIR = REPO_ROOT / "results" / "condition0"

logger = logging.getLogger(__name__)


class _SafeEncoder(json.JSONEncoder):
    """Encode dataclass instances and other non-serializable objects."""

    def default(self, o):
        if hasattr(o, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(o)
        return super().default(o)


def _serialize_constraints(constraints):
    """Convert ConstraintInfo objects to plain dicts for JSON storage."""
    if not constraints:
        return []
    from dataclasses import asdict
    return [asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in constraints]



# ---------------------------------------------------------------------------
# Completed-task detection
# ---------------------------------------------------------------------------

def _get_completed_task_ids(results_dir: Path) -> set:
    """Scan results directory for completed runs and return their task_ids."""
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
            path = CONDITION0_TASKS_DIR / path.name
        if not path.exists():
            raise SystemExit(f"Task file not found: {path}")
        task = _load_task(path)
        task["max_steps_per_subgoal"] = args.max_steps_per_subgoal or task["max_steps_per_subgoal"]
        task["diary_check_interval"] = args.diary_check_interval or task["diary_check_interval"]
        task["max_corrections"] = args.max_corrections or task["max_corrections"]
        return [task]

    CONDITION0_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(glob.glob(str(CONDITION0_TASKS_DIR / "*.json")))
    if not json_files:
        raise SystemExit(f"No JSON files found in {CONDITION0_TASKS_DIR}")
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
# Interactive help prompt
# ---------------------------------------------------------------------------

def _ask_user_for_help(
    subgoal_nl: str,
    completion_pct: float,
    current_instruction: str,
    reasoning: str = "",
) -> tuple:
    """Prompt user for help when the system is stuck.

    Returns (choice, value) where choice is one of:
      "correction"       - new OpenVLA instruction, same subgoal (value = instruction)
      "override_subgoal" - new subgoal text (value = subgoal)
      "replan"           - new high-level NL instruction for full LTL replanning (value = instruction)
      "skip"             - skip this subgoal
      "abort"            - abort the entire mission
    """
    print(f"\n{'='*60}")
    print("SYSTEM REQUESTING HELP")
    print(f"  Subgoal: {subgoal_nl}")
    print(f"  Completion: {completion_pct:.0%}")
    print(f"  Current instruction: {current_instruction}")
    if reasoning:
        print(f"  Reasoning: {reasoning}")
    print(f"{'='*60}")
    print("[a] Provide a correction (new OpenVLA instruction, same subgoal)")
    print("[b] Override current subgoal")
    print("[c] Override entire plan (new high-level instruction)")
    print("[d] Skip this subgoal")
    print("[e] Abort mission")
    while True:
        choice = input("Choice [a/b/c/d/e]: ").strip().lower()
        if choice == "a":
            instr = input("New OpenVLA instruction: ").strip()
            if not instr:
                print("Empty instruction, please try again.")
                continue
            return ("correction", instr)
        elif choice == "b":
            subgoal = input("New subgoal: ").strip()
            if not subgoal:
                print("Empty subgoal, please try again.")
                continue
            return ("override_subgoal", subgoal)
        elif choice == "c":
            instr = input("New high-level instruction: ").strip()
            if not instr:
                print("Empty instruction, please try again.")
                continue
            return ("replan", instr)
        elif choice == "d":
            return ("skip", "")
        elif choice == "e":
            return ("abort", "")
        else:
            print("Invalid choice, please enter a, b, c, d, or e.")


# ---------------------------------------------------------------------------
# Per-subgoal control loop
# ---------------------------------------------------------------------------

def _run_subgoal(
    env: Any,
    batch: Any,
    server_url: str,
    subgoal_nl: str,
    monitor_model: str,
    llm_model: str,
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
    frame_offset: int,
    trajectory_log: List[Dict[str, Any]],
    check_interval_s: Optional[float] = None,
    max_seconds: Optional[float] = None,
    constraints: Optional[List] = None,
) -> Dict[str, Any]:
    """Run the goal-adherence-monitored control loop for a single subgoal.

    Returns a dict with subgoal-level results and the final world-space origin
    for the next subgoal.
    """
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

    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    openvla_pose_origin: List[float] = [0.0, 0.0, 0.0, 0.0]
    last_pose: Optional[List[float]] = None
    small_count = 0
    override_history: List[Dict[str, Any]] = []
    in_correction = False
    last_correction_step = -check_interval
    last_correction_time = time.time() if use_async else None
    subgoal_start_time = time.time()
    stop_reason = "max_steps"
    total_steps = 0
    replan_instruction = ""

    batch.reset_model(server_url)

    cam_id = drone_cam_id
    result = None
    step = 0

    def _process_help(completion_pct: float, reasoning: str, trigger: str) -> str:
        """Handle interactive help prompt. Returns "retry" or "break"."""
        nonlocal current_instruction, subgoal_nl, converted_instruction
        nonlocal openvla_pose_origin, small_count, last_pose
        nonlocal in_correction, last_correction_step, last_correction_time
        nonlocal monitor, converter
        nonlocal stop_reason, total_steps, replan_instruction

        logger.warning(
            "Ask-help triggered at step %d by %s (completion: %.0f%%): %s",
            step, trigger, completion_pct * 100, reasoning,
        )

        choice, value = _ask_user_for_help(
            subgoal_nl, completion_pct, current_instruction, reasoning,
        )

        logger.info(
            "User chose '%s' at step %d (value: %s)",
            choice, step, repr(value) if value else "(none)",
        )

        if choice == "correction":
            old_instruction = current_instruction
            current_instruction = value
            openvla_pose_origin = list(current_pose)
            small_count = 0
            last_pose = None
            in_correction = True
            last_correction_step = step
            if use_async:
                last_correction_time = time.time()
            batch.reset_model(server_url)
            override_history.append({
                "step": step,
                "type": "user_correction",
                "trigger": trigger,
                "old_instruction": old_instruction,
                "new_instruction": value,
            })
            logger.info(
                "User correction applied: '%s' -> '%s' (subgoal unchanged: '%s')",
                old_instruction, value, subgoal_nl,
            )
            return "retry"

        if choice == "override_subgoal":
            monitor.cleanup()
            old_subgoal = subgoal_nl
            subgoal_nl = value
            logger.info(
                "User overriding subgoal: '%s' -> '%s'. Running SubgoalConverter...",
                old_subgoal, value,
            )
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
            openvla_pose_origin = list(current_pose)
            small_count = 0
            last_pose = None
            in_correction = False
            last_correction_step = step
            if use_async:
                last_correction_time = time.time()
            batch.reset_model(server_url)
            override_history.append({
                "step": step,
                "type": "user_override_subgoal",
                "trigger": trigger,
                "old_subgoal": old_subgoal,
                "new_subgoal": value,
                "new_instruction": converted_instruction,
            })
            logger.info(
                "Subgoal overridden: '%s' -> '%s' (OpenVLA instruction: '%s')",
                old_subgoal, value, converted_instruction,
            )
            return "retry"

        if choice == "replan":
            logger.info(
                "User requesting full replan with new instruction: '%s'", value,
            )
            stop_reason = "replan"
            replan_instruction = value
            total_steps = step
            return "break"

        if choice == "abort":
            logger.info("User aborted mission at step %d.", step)
            stop_reason = "abort"
            total_steps = step
            return "break"

        # skip
        logger.info("User skipped subgoal '%s' at step %d.", subgoal_nl, step)
        stop_reason = "skipped"
        total_steps = step
        return "break"

    while step < max_steps:
        if max_seconds is not None and (time.time() - subgoal_start_time) >= max_seconds:
            logger.info("Max seconds (%.1f) reached at step %d.", max_seconds, step)
            stop_reason = "max_seconds"
            total_steps = step
            break

        if use_async:
            async_result = monitor.poll_result()
            if async_result is not None:
                if async_result.action == "stop":
                    logger.info("Async monitor stop at step %d: %s", step, async_result.reasoning)
                    stop_reason = "monitor_complete"
                    total_steps = step
                    break
                if async_result.action == "ask_help":
                    if _process_help(async_result.completion_pct, async_result.reasoning, "async_monitor") == "retry":
                        continue
                    break
                if async_result.action == "force_converge":
                    logger.info("Async monitor force_converge at step %d: %s", step, async_result.reasoning)
                    override_history.append({
                        "step": step,
                        "type": "force_converge",
                        "reasoning": async_result.reasoning,
                        "constraint_violated": async_result.constraint_violated,
                    })

        image = set_drone_cam_and_get_image(env, cam_id)
        if image is None:
            logger.warning("No image at step %d, ending subgoal.", step)
            stop_reason = "no_image"
            break

        global_frame_idx = frame_offset + step
        frame_path = frames_dir / f"frame_{global_frame_idx:06d}.png"
        try:
            import cv2
            cv2.imwrite(str(frame_path), image)
        except Exception as e:
            logger.debug("Failed to save frame %s: %s", frame_path, e)

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
                logger.info("Monitor stop at step %d: %s", step, result.reasoning)
                stop_reason = "monitor_complete"
                total_steps = step
                break
            if result.action == "ask_help":
                if _process_help(result.completion_pct, result.reasoning, "sync_monitor") == "retry":
                    continue
                break
            if result.action == "force_converge":
                logger.info("Monitor force_converge at step %d: %s", step, result.reasoning)
                override_history.append({
                    "step": step,
                    "type": "force_converge",
                    "reasoning": result.reasoning,
                    "constraint_violated": result.constraint_violated,
                })

        openvla_pose = [c - o for c, o in zip(current_pose, openvla_pose_origin)]
        response = batch.send_prediction_request(
            image=Image.fromarray(image),
            proprio=state_for_openvla(openvla_pose),
            instr=current_instruction.strip().lower(),
            server_url=server_url,
        )

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
        if not use_async:
            if result is not None and result.action == "force_converge":
                converged = True
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
            conv_frame = set_drone_cam_and_get_image(env, cam_id)
            if conv_frame is not None:
                conv_path = frames_dir / f"frame_conv_{global_frame_idx:06d}.png"
                try:
                    import cv2
                    cv2.imwrite(str(conv_path), conv_frame)
                except Exception:
                    conv_path = frame_path
            else:
                conv_path = frame_path

            try:
                conv_result = monitor.on_convergence(
                    conv_path, displacement=list(current_pose),
                )
            except Exception as e:
                logger.error("monitor.on_convergence failed at step %d: %s", step, e)
                conv_result = DiaryCheckResult(
                    action="stop", new_instruction="",
                    reasoning="LLM error on convergence",
                    diary_entry="",
                    completion_pct=monitor.last_completion_pct,
                )

            if conv_result.action == "stop":
                logger.info(
                    "Subgoal complete on convergence at step %d: %s",
                    step, conv_result.reasoning,
                )
                in_correction = False
                stop_reason = "monitor_complete"
                break

            elif conv_result.action == "ask_help":
                in_correction = False
                help_decision = _process_help(
                    conv_result.completion_pct, conv_result.reasoning, "convergence",
                )
                if help_decision == "break":
                    break
                # "retry": fall through to step += 1

            elif conv_result.new_instruction:
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
                if use_async:
                    last_correction_time = time.time()
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

    all_vlm_call_records = list(monitor.vlm_rtts) + list(converter.llm_call_records)
    constraint_violation_count = sum(
        1 for o in override_history
        if o.get("constraint_violated", False)
    )

    # Save diary summary for this subgoal
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
        "in_correction_at_end": in_correction,
        "constraint_violation_count": constraint_violation_count,
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
        "peak_completion": monitor.peak_completion,
        "vlm_call_count": monitor.vlm_calls,
        "vlm_call_records": all_vlm_call_records,
        "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
        "constraints": _serialize_constraints(constraints),
        "constraint_violation_count": constraint_violation_count,
        "parse_failures": monitor.parse_failures,
        "replan_instruction": replan_instruction,
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
    env.teleport(initial_pos[0:3], initial_pos[4])
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

    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[4]

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
                safe_name = _sanitize_name(current_subgoal)
                subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"

                active_constraints = planner.get_active_constraints()
                if active_constraints:
                    logger.info(
                        "Active constraints for subgoal %d: %s",
                        subgoal_index, active_constraints,
                    )

                logger.info(
                    "--- Subgoal %d: '%s' ---", subgoal_index, current_subgoal,
                )

                try:
                    subgoal_result = _run_subgoal(
                        env=env,
                        batch=batch,
                        server_url=server_url,
                        subgoal_nl=current_subgoal,
                        monitor_model=monitor_model,
                        llm_model=llm_model,
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
                        frame_offset=total_frame_count,
                        trajectory_log=trajectory_log,
                        check_interval_s=check_interval_s,
                        max_seconds=max_seconds,
                        constraints=active_constraints,
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

                if sr == "abort":
                    logger.info("Mission aborted by user.")
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
        any_constraint_violated = any(
            s.get("constraint_violation_count", 0) > 0 for s in subgoal_summaries
        )
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
            "subgoal_count": subgoal_index,
            "subgoal_summaries": subgoal_summaries,
            "total_steps": total_steps_all,
            "total_vlm_calls": total_vlm_calls,
            "total_corrections": total_corrections,
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
        default=DEFAULT_INITIAL_POSITION,
        metavar="x,y,z,yaw",
        help="Initial position (for -c mode). Default: %(default)s",
    )

    # Sim / server
    parser.add_argument("-e", "--env_id", default=DOWNTOWN_ENV_ID, help="Environment ID")
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
            logger.info("Camera selection: pick the camera for OpenVLA.")
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
            logger.info(
                "\n===== Task %d/%d: '%s' =====",
                idx + 1, len(tasks), task["instruction"][:80],
            )
            ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            task_label = task.get("task_id") or _sanitize_name(task["instruction"], max_len=30)
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
                    env_id=args.env_id,
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
