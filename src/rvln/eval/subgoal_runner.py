"""
Centralized subgoal execution engine with configurable monitoring.

All condition scripts delegate to run_subgoal() with appropriate flags.
This eliminates duplication and ensures consistent behavior/logging
across ablation conditions.
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from PIL import Image

from rvln.config import (
    ACTION_SMALL_DELTA_POS,
    ACTION_SMALL_DELTA_YAW,
    DEFAULT_DIARY_CHECK_INTERVAL,
    DEFAULT_MAX_CORRECTIONS,
    DEFAULT_MAX_STEPS_PER_SUBGOAL,
    DEFAULT_VLM_MODEL,
)

logger = logging.getLogger(__name__)

MonitorMode = Literal["full", "single_frame", "grid_only", "text_only", "none"]


@dataclass
class SubgoalConfig:
    """Configuration flags for how a subgoal is executed and monitored."""
    monitor_mode: MonitorMode = "full"
    use_constraints: bool = True
    check_interval: int = DEFAULT_DIARY_CHECK_INTERVAL
    max_steps: int = DEFAULT_MAX_STEPS_PER_SUBGOAL
    max_corrections: int = DEFAULT_MAX_CORRECTIONS
    check_interval_s: Optional[float] = None
    max_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Prompt templates for ablation modes
# ---------------------------------------------------------------------------

GRID_ONLY_GLOBAL_PROMPT = """\
Subgoal: {subgoal}

Previous estimated completion: {prev_completion_pct}

The grid shows up to the 9 most recent sampled frames (left to right, top to
bottom, in temporal order).

Based on the visual progression in the grid of sampled frames, respond with
EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "should_stop": true/false,
  "reasoning": "..."
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished. Do NOT mark complete for partial progress.
- "completion_percentage": your best estimate of how close the subgoal is to
  completion (0.0 = not started, 1.0 = fully done). Reserve 1.0 for
  high-confidence completion. Use the full 0.0-0.99 range; pick the specific
  value you estimate rather than parking on a single round number.
- "should_stop": true if the drone appears off-track or heading toward a
  collision. The drone will be stopped and a corrective instruction issued.
  Do NOT set true for slow progress.
- "reasoning": one short sentence explaining your judgement, especially why
  should_stop was set to true. This text is forwarded to the convergence
  stage if the drone is stopped, so be specific."""

GRID_ONLY_GLOBAL_PROMPT_CONSTRAINTS = """\
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
  "should_stop": true/false,
  "reasoning": "..."
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished. Do NOT mark complete for partial progress.
- "completion_percentage": your best estimate of how close the subgoal is to
  completion (0.0 = not started, 1.0 = fully done). Reserve 1.0 for
  high-confidence completion. Use the full 0.0-0.99 range; pick the specific
  value you estimate rather than parking on a single round number.
- "should_stop": true if the drone appears off-track, heading toward a
  collision, or violating any active constraint listed above. The drone will
  be stopped and a corrective instruction issued.
  Do NOT set true for slow progress.
- "reasoning": one short sentence explaining your judgement, especially which
  constraint or hazard motivated should_stop when true. This text is
  forwarded to the convergence stage if the drone is stopped, so be
  specific."""

GRID_ONLY_CONVERGENCE_PROMPT = """\
Subgoal: {subgoal}

Previous estimated completion: {prev_completion_pct}
{stop_reasoning_block}
The drone has stopped moving. The grid shows up to the 9 most recent sampled
frames (left to right, top to bottom, in temporal order).

Given the visual progression in the sampled frames, is the subgoal complete?
If not, did the drone stop short or overshoot?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "diagnosis": "stopped_short" or "overshot" or "complete",
  "corrective_instruction": "..." or null
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished.
- "diagnosis": "complete" if done, "stopped_short" if the drone needs to keep
  going, "overshot" if the drone went past the goal.
- "corrective_instruction": REQUIRED if not complete. A single-action drone
  command. null only if complete."""

GRID_ONLY_CONVERGENCE_PROMPT_CONSTRAINTS = """\
Subgoal: {subgoal}
{constraints_block}
Previous estimated completion: {prev_completion_pct}
{stop_reasoning_block}
The drone has stopped moving. The grid shows up to the 9 most recent sampled
frames (left to right, top to bottom, in temporal order).

Given the visual progression in the sampled frames, is the subgoal complete?
If not, did the drone stop short or overshoot?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "diagnosis": "stopped_short" or "overshot" or "complete" or "constraint_violated",
  "corrective_instruction": "..." or null
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished.
- "diagnosis": "complete" if done, "stopped_short" if the drone needs to keep
  going, "overshot" if the drone went past the goal, "constraint_violated" if
  an active constraint was breached.
- "corrective_instruction": REQUIRED if not complete. A single-action drone
  command. If a constraint was violated, the corrective instruction should
  restore compliance: move away from a forbidden region for avoidance
  constraints, or restore the required condition for maintenance constraints
  (e.g., "ascend 2 meters" to regain an altitude constraint).
  null only if complete."""

SINGLE_FRAME_GLOBAL_PROMPT = """\
Subgoal: {subgoal}

Previous estimated completion: {prev_completion_pct}

You are looking at the drone's current camera view. Based on this single frame,
is the subgoal complete?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "should_stop": true/false,
  "reasoning": "..."
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished based on the current view.
- "completion_percentage": your best estimate (0.0 = not started, 1.0 = done).
  Reserve 1.0 for high-confidence completion. Use the full 0.0-0.99 range;
  pick the specific value you estimate rather than parking on a single value.
- "should_stop": true if the drone appears off-track or heading toward a
  collision. The drone will be stopped and a correction issued.
- "reasoning": one short sentence explaining your judgement, especially why
  should_stop was set to true. This text is forwarded to the convergence
  stage if the drone is stopped, so be specific."""

SINGLE_FRAME_GLOBAL_PROMPT_CONSTRAINTS = """\
Subgoal: {subgoal}
{constraints_block}
Previous estimated completion: {prev_completion_pct}

You are looking at the drone's current camera view. Based on this single frame,
is the subgoal complete?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "should_stop": true/false,
  "reasoning": "..."
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished based on the current view.
- "completion_percentage": your best estimate (0.0 = not started, 1.0 = done).
  Reserve 1.0 for high-confidence completion. Use the full 0.0-0.99 range;
  pick the specific value you estimate rather than parking on a single value.
- "should_stop": true if the drone appears off-track, heading toward a
  collision, or violating any active constraint. The drone will be stopped
  and a correction issued.
- "reasoning": one short sentence explaining your judgement, especially which
  constraint or hazard motivated should_stop when true. This text is
  forwarded to the convergence stage if the drone is stopped, so be
  specific."""

SINGLE_FRAME_CONVERGENCE_PROMPT = """\
Subgoal: {subgoal}

Previous estimated completion: {prev_completion_pct}
{stop_reasoning_block}
The drone has stopped moving. Based on the current camera view, is the subgoal
complete? If not, did it stop short or overshoot?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "diagnosis": "stopped_short" or "overshot" or "complete",
  "corrective_instruction": "..." or null
}}

- "complete": true ONLY if you are highly confident.
- "diagnosis": "complete" if done, "stopped_short" if needs more, "overshot"
  if past the goal.
- "corrective_instruction": REQUIRED if not complete. A single-action command."""

SINGLE_FRAME_CONVERGENCE_PROMPT_CONSTRAINTS = """\
Subgoal: {subgoal}
{constraints_block}
Previous estimated completion: {prev_completion_pct}
{stop_reasoning_block}
The drone has stopped moving. Based on the current camera view, is the subgoal
complete? If not, did it stop short or overshoot?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "diagnosis": "stopped_short" or "overshot" or "complete" or "constraint_violated",
  "corrective_instruction": "..." or null
}}

- "complete": true ONLY if you are highly confident.
- "diagnosis": "complete" if done, "stopped_short" if needs more, "overshot"
  if past the goal, "constraint_violated" if an active constraint was breached.
- "corrective_instruction": REQUIRED if not complete. A single-action command.
  If a constraint was violated, restore compliance: move away from a forbidden
  region for avoidance constraints, or restore the required condition for
  maintenance constraints (e.g., "ascend 2 meters")."""


def _patch_monitor_prompts(mode: MonitorMode) -> None:
    """Set GoalAdherenceMonitor prompt templates for the given mode.

    Must be called for every mode (including "full") to prevent stale
    templates from a previous call leaking into the next run.
    """
    import rvln.ai.goal_adherence_monitor as gam
    from rvln.ai.prompts import (
        DIARY_GLOBAL_PROMPT,
        DIARY_GLOBAL_PROMPT_WITH_CONSTRAINTS,
        DIARY_CONVERGENCE_PROMPT,
        DIARY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS,
    )

    if mode == "grid_only":
        gam.GLOBAL_PROMPT_TEMPLATE = GRID_ONLY_GLOBAL_PROMPT
        gam.GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS = GRID_ONLY_GLOBAL_PROMPT_CONSTRAINTS
        gam.CONVERGENCE_PROMPT_TEMPLATE = GRID_ONLY_CONVERGENCE_PROMPT
        gam.CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS = GRID_ONLY_CONVERGENCE_PROMPT_CONSTRAINTS
    elif mode == "single_frame":
        gam.GLOBAL_PROMPT_TEMPLATE = SINGLE_FRAME_GLOBAL_PROMPT
        gam.GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS = SINGLE_FRAME_GLOBAL_PROMPT_CONSTRAINTS
        gam.CONVERGENCE_PROMPT_TEMPLATE = SINGLE_FRAME_CONVERGENCE_PROMPT
        gam.CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS = SINGLE_FRAME_CONVERGENCE_PROMPT_CONSTRAINTS
    else:
        gam.GLOBAL_PROMPT_TEMPLATE = DIARY_GLOBAL_PROMPT
        gam.GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS = DIARY_GLOBAL_PROMPT_WITH_CONSTRAINTS
        gam.CONVERGENCE_PROMPT_TEMPLATE = DIARY_CONVERGENCE_PROMPT
        gam.CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS = DIARY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS


# ---------------------------------------------------------------------------
# Core subgoal execution
# ---------------------------------------------------------------------------

def run_subgoal(
    env: Any,
    batch: Any,
    server_url: str,
    subgoal_nl: str,
    monitor_model: str,
    llm_model: str,
    config: SubgoalConfig,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    origin_yaw: float,
    drone_cam_id: int,
    frames_dir: Path,
    subgoal_dir: Path,
    frame_offset: int,
    trajectory_log: List[Dict[str, Any]],
    constraints: Optional[List] = None,
    ask_help_callback=None,
) -> Dict[str, Any]:
    """Run the control loop for a single subgoal with configurable monitoring.

    Parameters
    ----------
    env : simulation environment
    batch : batch module with send_prediction_request, reset_model, etc.
    server_url : OpenVLA server URL
    subgoal_nl : natural language subgoal description
    monitor_model : VLM model name for monitoring
    llm_model : LLM model name for subgoal conversion
    config : SubgoalConfig with monitoring flags
    origin_x, origin_y, origin_z, origin_yaw : world-space origin
    drone_cam_id : camera ID
    frames_dir : directory for saving frames
    subgoal_dir : directory for subgoal-specific artifacts
    frame_offset : global frame index offset
    trajectory_log : shared trajectory log (appended to)
    constraints : list of ConstraintInfo objects (or None)
    ask_help_callback : optional callable(subgoal_nl, completion_pct, instruction, reasoning)
        returning (choice, value). If None, ask_help triggers stop.

    Returns dict with subgoal-level results.
    """
    from rvln.ai.goal_adherence_monitor import DiaryCheckResult, GoalAdherenceMonitor
    from rvln.ai.subgoal_converter import SubgoalConverter
    from rvln.sim.env_setup import (
        apply_action_poses,
        relative_pose_to_world,
        set_drone_cam_and_get_image,
        state_for_openvla,
    )

    _patch_monitor_prompts(config.monitor_mode)

    effective_constraints = constraints if config.use_constraints else None

    use_async = config.check_interval_s is not None

    subgoal_dir.mkdir(parents=True, exist_ok=True)
    diary_artifacts = subgoal_dir / "diary_artifacts"
    diary_artifacts.mkdir(parents=True, exist_ok=True)

    converter = SubgoalConverter(model=llm_model)
    conversion = converter.convert(subgoal_nl)
    converted_instruction = conversion.instruction
    current_instruction = converted_instruction

    use_monitor = config.monitor_mode != "none"

    extra_kwargs: Dict[str, Any] = {}
    if config.monitor_mode == "text_only":
        extra_kwargs["global_backend"] = "text_llm"
        extra_kwargs["global_model"] = llm_model
    if config.monitor_mode == "single_frame":
        extra_kwargs["single_frame_mode"] = True

    monitor: Optional[GoalAdherenceMonitor] = None
    if use_monitor:
        monitor = GoalAdherenceMonitor(
            subgoal=subgoal_nl,
            check_interval=config.check_interval,
            model=monitor_model,
            artifacts_dir=diary_artifacts,
            max_corrections=config.max_corrections,
            check_interval_s=config.check_interval_s,
            constraints=effective_constraints,
            **extra_kwargs,
        )

    current_pose: List[float] = [0.0, 0.0, 0.0, 0.0]
    openvla_pose_origin: List[float] = [0.0, 0.0, 0.0, 0.0]
    last_pose: Optional[List[float]] = None
    small_count = 0
    override_history: List[Dict[str, Any]] = []
    in_correction = False
    last_correction_step = -config.check_interval
    last_correction_time = time.time() if use_async else None
    subgoal_start_time = time.time()
    # Default terminal reason if no other branch sets it. Reaching the step
    # ceiling is treated as an implicit ask-for-help: the agent has had its
    # full budget and not converged, which (per experiment policy) aborts the
    # mission and is counted as an ask-for-help event.
    stop_reason = "ask_help"
    total_steps = 0
    replan_instruction = ""

    batch.reset_model(server_url)

    cam_id = drone_cam_id
    result = None
    step = 0

    from rvln.eval.step_timer import StepTimer
    _step_timer = StepTimer(frames_dir.parent / "step_timings.jsonl")

    def _process_help(completion_pct: float, reasoning: str, trigger: str) -> str:
        nonlocal current_instruction, subgoal_nl, converted_instruction
        nonlocal openvla_pose_origin, small_count, last_pose
        nonlocal in_correction, last_correction_step, last_correction_time
        nonlocal monitor, converter
        nonlocal stop_reason, total_steps, replan_instruction

        logger.warning(
            "Ask-help triggered at step %d by %s (completion: %.0f%%): %s",
            step, trigger, completion_pct * 100, reasoning,
        )

        # Max corrections exhausted is a hard ceiling, just like max_steps:
        # the agent has used its full correction budget without converging.
        # Abort the mission directly (counted as ask_help) instead of asking
        # the callback, which could otherwise issue a correction we can't honor.
        if monitor is not None and monitor.corrections_exhausted:
            logger.warning(
                "Max corrections (%d) exhausted for subgoal '%s'; "
                "treating as ask_help and aborting mission.",
                monitor.max_corrections, subgoal_nl,
            )
            stop_reason = "ask_help"
            total_steps = step
            return "break"

        if ask_help_callback is None:
            stop_reason = "ask_help_no_handler"
            total_steps = step
            return "break"

        choice, value = ask_help_callback(
            subgoal_nl, completion_pct, current_instruction, reasoning,
        )

        logger.info("User chose '%s' at step %d", choice, step)

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
            return "retry"

        if choice == "override_subgoal":
            if monitor:
                monitor.cleanup()
            old_subgoal = subgoal_nl
            subgoal_nl = value
            converter = SubgoalConverter(model=llm_model)
            conversion = converter.convert(subgoal_nl)
            converted_instruction = conversion.instruction
            current_instruction = converted_instruction
            if use_monitor:
                monitor = GoalAdherenceMonitor(
                    subgoal=subgoal_nl,
                    check_interval=config.check_interval,
                    model=monitor_model,
                    artifacts_dir=diary_artifacts,
                    max_corrections=config.max_corrections,
                    check_interval_s=config.check_interval_s,
                    constraints=effective_constraints,
                    **extra_kwargs,
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
            return "retry"

        if choice == "replan":
            stop_reason = "replan"
            replan_instruction = value
            total_steps = step
            return "break"

        if choice == "abort":
            # Counted as an ask-for-help event (the trigger that led to this
            # abort was an ask_help). Episode-level runners treat "ask_help"
            # the same as "abort": they end the mission immediately.
            stop_reason = "ask_help"
            total_steps = step
            return "break"

        # skip
        stop_reason = "skipped"
        total_steps = step
        return "break"

    # Loop-carried frame: first iteration fetches via /get_frame, subsequent
    # iterations reuse the image returned by /step.
    image: Optional[np.ndarray] = None

    while step < config.max_steps:
        _step_timer.start_step(step)
        try:
            if config.max_seconds is not None and (time.time() - subgoal_start_time) >= config.max_seconds:
                logger.info("Max seconds (%.1f) reached at step %d.", config.max_seconds, step)
                stop_reason = "max_seconds"
                total_steps = step
                break

            async_force_converge = False
            if use_async and monitor:
                async_result = monitor.poll_result()
                if async_result is not None:
                    if async_result.action == "stop":
                        stop_reason = "monitor_complete"
                        total_steps = step
                        break
                    if async_result.action == "ask_help":
                        if _process_help(async_result.completion_pct, async_result.reasoning, "async_monitor") == "retry":
                            continue
                        break
                    if async_result.action == "force_converge":
                        override_history.append({
                            "step": step,
                            "type": "force_converge",
                            "reasoning": async_result.reasoning,
                        })
                        async_force_converge = True

            if image is None:
                with _step_timer.phase("get_frame"):
                    image = set_drone_cam_and_get_image(env, cam_id)
                if image is None:
                    logger.warning("No image at step %d, ending subgoal.", step)
                    stop_reason = "no_image"
                    break

            global_frame_idx = frame_offset + step
            frame_path = frames_dir / f"frame_{global_frame_idx:06d}.png"
            with _step_timer.phase("frame_write"):
                try:
                    import cv2
                    cv2.imwrite(str(frame_path), image)
                except Exception as e:
                    logger.debug("Failed to save frame %s: %s", frame_path, e)

            with _step_timer.phase("monitor_on_frame"):
                if monitor:
                    try:
                        result = monitor.on_frame(frame_path, displacement=list(current_pose))
                    except Exception as e:
                        logger.error("monitor.on_frame failed at step %d: %s", step, e)
                        result = DiaryCheckResult(
                            action="continue", new_instruction="", reasoning="",
                            diary_entry="", completion_pct=monitor.last_completion_pct,
                        )

            if monitor and not use_async:
                if result.action == "stop":
                    stop_reason = "monitor_complete"
                    total_steps = step
                    break
                if result.action == "ask_help":
                    if _process_help(result.completion_pct, result.reasoning, "sync_monitor") == "retry":
                        continue
                    break
                if result.action == "force_converge":
                    override_history.append({
                        "step": step,
                        "type": "force_converge",
                        "reasoning": result.reasoning,
                    })

            openvla_pose = [c - o for c, o in zip(current_pose, openvla_pose_origin)]
            with _step_timer.phase("predict"):
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
                        sleep_s=0.1,
                        drone_cam_id=cam_id,
                    )
                except Exception as e:
                    logger.error("Error executing action at step %d: %s", step, e)
                    stop_reason = "action_error"
                    total_steps = step
                    break

            total_steps = step + 1

            if new_image is None:
                # /step didn't return an image; fall back to /get_frame next iter.
                image = None
            else:
                image = new_image

            # --- Convergence detection ---
            # Note: checkpoints (monitor.on_frame above) still fire during
            # corrective instruction execution, but they evaluate the *subgoal*,
            # not the corrective micro-command. The convergence guard below
            # (steps_since_correction >= check_interval) is the only mechanism
            # that distinguishes correction-phase from normal execution: it
            # suppresses the small-motion convergence detector to let the new
            # instruction take effect. See the "Correction-awareness gap"
            # section in goal_adherence_monitor.py for details.
            converged = False
            if not use_async:
                if monitor and result is not None and result.action == "force_converge":
                    converged = True
                steps_since_correction = step - last_correction_step
                if last_pose is not None and steps_since_correction >= config.check_interval:
                    diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
                    if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                        small_count += 1
                    else:
                        small_count = 0
                    if small_count >= batch.ACTION_SMALL_STEPS:
                        converged = True
            else:
                if async_force_converge:
                    converged = True
                elapsed_since_correction = time.time() - last_correction_time
                if last_pose is not None and elapsed_since_correction >= config.check_interval_s:
                    diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
                    if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                        small_count += 1
                    else:
                        small_count = 0
                    if small_count >= batch.ACTION_SMALL_STEPS:
                        converged = True
            last_pose = list(current_pose)

            if converged:
                if not monitor:
                    # No monitor (C3 open-loop): convergence = done
                    stop_reason = "convergence"
                    total_steps = step + 1
                    break

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

                elif conv_result.new_instruction:
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
                    in_correction = False
                    stop_reason = "convergence_no_command"
                    break

            step += 1
        finally:
            _step_timer.end_step()
    else:
        # Reached the step ceiling without completing. Counted as an
        # ask-for-help event and aborts the mission at the episode level.
        logger.warning(
            "Max step count (%d) reached for subgoal '%s'; "
            "treating as ask_help and aborting mission.",
            config.max_steps, subgoal_nl,
        )
        stop_reason = "ask_help"
        total_steps = config.max_steps

    _step_timer.close()

    # Collect metrics
    all_vlm_call_records = []
    if monitor:
        all_vlm_call_records = list(monitor.vlm_rtts)
    all_vlm_call_records += list(converter.llm_call_records)

    diary_summary = {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "diary": monitor.diary if monitor else [],
        "override_history": override_history,
        "corrections_used": monitor.corrections_used if monitor else 0,
        "last_completion_pct": monitor.last_completion_pct if monitor else 0.0,
        "peak_completion": monitor.peak_completion if monitor else 0.0,
        "parse_failures": monitor.parse_failures if monitor else 0,
        "vlm_call_count": monitor.vlm_calls if monitor else 0,
        "vlm_call_records": list(monitor.vlm_rtts) if monitor else [],
        "converter_call_records": list(converter.llm_call_records),
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "in_correction_at_end": in_correction,
    }
    with open(subgoal_dir / "diary_summary.json", "w") as f:
        json.dump(diary_summary, f, indent=2)

    if monitor:
        monitor.cleanup()

    next_origin_x, next_origin_y, next_origin_z, next_origin_yaw = relative_pose_to_world(
        origin_x, origin_y, origin_z, origin_yaw, current_pose,
    )

    return {
        "subgoal": subgoal_nl,
        "converted_instruction": converted_instruction,
        "total_steps": total_steps,
        "stop_reason": stop_reason,
        "corrections_used": monitor.corrections_used if monitor else 0,
        "last_completion_pct": monitor.last_completion_pct if monitor else 0.0,
        "peak_completion": monitor.peak_completion if monitor else 0.0,
        "vlm_call_count": monitor.vlm_calls if monitor else 0,
        "vlm_call_records": all_vlm_call_records,
        "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
        "constraints": _serialize_constraints(effective_constraints),
        "parse_failures": monitor.parse_failures if monitor else 0,
        "replan_instruction": replan_instruction,
    }


def _serialize_constraints(constraints):
    if not constraints:
        return []
    from dataclasses import asdict
    return [asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in constraints]
