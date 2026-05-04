#!/usr/bin/env python3
"""
Condition 2: LLM Sequential Planner (No LTL).

Uses an LLM to decompose the instruction into a numbered list of subgoals
(text-to-text, no LTL formula, no Spot automaton). Executes subgoals
sequentially with full goal adherence monitoring (same as Condition 0). No formal
constraint enforcement.

OpenVLA server must be running: python scripts/start_server.py

Usage (from repo root):
  python scripts/run_condition2_llm_planner.py --task third_task.json
  python scripts/run_condition2_llm_planner.py --run_all_tasks
  python scripts/run_condition2_llm_planner.py -c "Go to the tree then land" --initial-position -181,7331,876,-89
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
from rvln.eval.subgoal_runner import SubgoalConfig, run_subgoal
from rvln.paths import (
    BATCH_SCRIPT,
    REPO_ROOT,
    UAV_FLOW_EVAL,
)
from rvln.maps import validate_task_map
from rvln.sim.env_setup import (
    import_batch_module,
    interactive_camera_select,
    load_env_vars,
    normalize_initial_pos,
    parse_position,
    setup_sim_env,
)

SHARED_TASKS_DIR = REPO_ROOT / "tasks"
CONDITION2_RESULTS_DIR = REPO_ROOT / "results" / "condition2"

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


LLM_DECOMPOSITION_PROMPT = """\
You are a task planner for an autonomous drone. Given a complex, multi-step
natural language instruction, decompose it into a sequential list of simple
subgoals that the drone should execute one at a time.

Rules:
- Each subgoal should be a single, concrete navigation action (e.g., "go to the tree",
  "turn right until you see the building", "approach the streetlight").
- Preserve the order specified in the instruction.
- If the instruction mentions constraints like "never fly over X" or "stay away
  from Y", you may mention them as reminders in the subgoal text, but you cannot
  formally enforce them.
- Output EXACTLY ONE JSON object with a "subgoals" key containing a list of strings.

Example input: "Go to the tree, then the streetlight, but never fly over the building."
Example output:
{
  "subgoals": [
    "Go to the tree (avoid flying over the building)",
    "Go to the streetlight (avoid flying over the building)"
  ]
}"""


def _llm_decompose(instruction: str, llm_model: str):
    """Decompose instruction into subgoals using a plain LLM call.

    Returns (subgoals, call_record) where call_record is a dict with
    rtt_s, model, input_tokens, output_tokens.
    """
    import time as _time
    from rvln.ai.utils.llm_providers import LLMFactory

    llm = LLMFactory.create(model=llm_model)
    messages = [
        {"role": "system", "content": LLM_DECOMPOSITION_PROMPT},
        {"role": "user", "content": instruction},
    ]
    t0 = _time.time()
    response = llm.make_request(messages, temperature=0.0, json_mode=True)
    rtt = _time.time() - t0
    usage = llm.last_usage
    call_record = {
        "label": "llm_decomposition",
        "rtt_s": round(rtt, 3),
        "model": usage.get("model", llm_model),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(response[start:end + 1])
        else:
            raise ValueError(f"LLM decomposition returned unparseable response: {response[:200]}")

    subgoals = parsed.get("subgoals", [])
    if not subgoals or not isinstance(subgoals, list):
        raise ValueError(f"LLM decomposition returned no subgoals: {parsed}")
    return [str(s).strip() for s in subgoals if str(s).strip()], call_record


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

    tasks_dir = SHARED_TASKS_DIR / map_info.task_dir_name

    if task_file is not None:
        validate_task_map(task_file, map_info)
        path = Path(task_file)
        if not path.is_absolute():
            if len(path.parts) > 1:
                path = SHARED_TASKS_DIR / path
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
      "replan"           - new high-level NL instruction for full replanning (value = instruction)
      "skip"             - skip this subgoal
      "abort"            - abort the entire mission

    In non-interactive mode (stdin is not a TTY), automatically skips.
    """
    if not sys.stdin.isatty():
        logger.warning(
            "ask_help triggered in non-interactive mode, auto-skipping subgoal '%s' "
            "(completion: %.0f%%, reason: %s)",
            subgoal_nl, completion_pct * 100, reasoning,
        )
        return ("skip", "")

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


def run_llm_planner_control_loop(
    env, batch, task, server_url, run_dir,
    llm_model, monitor_model, drone_cam_id,
    save_mp4=False, mp4_fps=10.0,
    check_interval_s=None, max_seconds=None,
    seed=DEFAULT_SEED, time_dilation=DEFAULT_TIME_DILATION,
    env_id="", diary_mode=DEFAULT_DIARY_MODE,
):
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

    # --- Decomposition and execution (with replan support) ---
    subgoal_summaries = []
    trajectory_log = []
    decomposition_records = []
    total_frame_count = 0
    subgoal_index = 0
    replan_count = 0
    aborted = False

    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[4]

    while True:
        logger.info(
            "%sLLM decomposition for: '%s'",
            f"[REPLAN #{replan_count}] " if replan_count > 0 else "",
            instruction,
        )
        subgoals, decomposition_call_record = _llm_decompose(instruction, llm_model)
        decomposition_records.append({
            "replan_index": replan_count,
            "instruction": instruction,
            "subgoals": subgoals,
            "call_record": decomposition_call_record,
        })
        logger.info("LLM subgoals: %s", subgoals)

        if not subgoals:
            logger.error("LLM decomposition produced no subgoals for: '%s'", instruction)
            if replan_count > 0:
                logger.warning("Replan produced no subgoals, ending run.")
                break
            raise ValueError("LLM decomposition produced no subgoals.")

        replan_requested = False

        for sg_i, subgoal_nl in enumerate(subgoals, 1):
            subgoal_index += 1
            safe_name = _sanitize_name(subgoal_nl)
            subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"

            logger.info(
                "--- Subgoal %d (plan step %d/%d): '%s' ---",
                subgoal_index, sg_i, len(subgoals), subgoal_nl,
            )

            sg_config = SubgoalConfig(
                monitor_mode="full",
                use_constraints=False,
                check_interval=check_interval,
                max_steps=max_steps_per_subgoal,
                max_corrections=max_corrections,
                check_interval_s=check_interval_s,
                max_seconds=max_seconds,
            )
            subgoal_result = run_subgoal(
                env=env, batch=batch, server_url=server_url,
                subgoal_nl=subgoal_nl, monitor_model=monitor_model, llm_model=llm_model,
                config=sg_config,
                origin_x=origin_x, origin_y=origin_y,
                origin_z=origin_z, origin_yaw=origin_yaw,
                drone_cam_id=drone_cam_id, frames_dir=frames_dir,
                subgoal_dir=subgoal_dir, frame_offset=total_frame_count,
                trajectory_log=trajectory_log,
                ask_help_callback=_ask_user_for_help,
            )

            total_frame_count += subgoal_result["total_steps"]
            subgoal_summaries.append(subgoal_result)

            sr = subgoal_result["stop_reason"]
            logger.info(
                "Subgoal %d finished: stop_reason=%s, steps=%d, completion=%.2f",
                subgoal_index, sr,
                subgoal_result["total_steps"],
                subgoal_result.get("last_completion_pct", 0.0),
            )

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
                logger.info("Subgoal '%s' skipped by user.", subgoal_nl)

        if aborted or not replan_requested:
            break

    logger.info(
        "Run finished: %d subgoals processed, %d replan(s), aborted=%s.",
        subgoal_index, replan_count, aborted,
    )

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

    total_steps_all = sum(s["total_steps"] for s in subgoal_summaries)
    total_vlm_calls = sum(s.get("vlm_call_count", 0) for s in subgoal_summaries)
    total_corrections = sum(s.get("corrections_used", 0) for s in subgoal_summaries)
    all_vlm_records = [r["call_record"] for r in decomposition_records]
    for s in subgoal_summaries:
        all_vlm_records.extend(s.get("vlm_call_records", []))
    total_input_tokens = sum(r.get("input_tokens", 0) for r in all_vlm_records)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in all_vlm_records)
    end_dt = datetime.fromisoformat(end_ts)
    start_dt = datetime.fromisoformat(start_ts)
    wall_clock_seconds = (end_dt - start_dt).total_seconds()

    run_info = {
        "condition": "condition2_llm_planner",
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
            "llm_decomposition": llm_model,
            "subgoal_converter": llm_model,
            "goal_adherence_monitor": monitor_model,
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
        "llm_subgoals": decomposition_records[0]["subgoals"] if decomposition_records else [],
        "decomposition_call_record": decomposition_records[0]["call_record"] if decomposition_records else {},
        "decomposition_records": decomposition_records,
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
        "any_constraint_violated": None,
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
        description="Condition 2: LLM sequential planner (no LTL) + full goal adherence monitoring",
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
    parser.add_argument("-o", "--results_dir", default=str(CONDITION2_RESULTS_DIR))
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
            run_name = f"c2_llm_planner__{task_label}__{ts}"
            try:
                run_dir = results_base / run_name
                use_time_mode = args.diary_mode == "time"
                run_info = run_llm_planner_control_loop(
                    env=env, batch=batch, task=task, server_url=server_url,
                    run_dir=run_dir, llm_model=args.llm_model, monitor_model=args.monitor_model,
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
    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting.")


if __name__ == "__main__":
    main()
