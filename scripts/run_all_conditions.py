#!/usr/bin/env python3
"""
Run all conditions back-to-back, optionally across multiple maps.

Tasks are loaded from a SHARED directory: tasks/<map_dir>/
Results go to: results/condition<N>/<map_dir>/<run_name>/

The orchestrator manages the simulator lifecycle automatically: it starts the
simulator for each map, runs all conditions, stops it, and proceeds to the next.

  --map filters to a single map.
  --conditions filters to specific conditions.
  --sim-controller delegates simulator management to a remote machine.

Usage (local, single machine):
  python scripts/run_all_conditions.py                          # all maps, all conditions
  python scripts/run_all_conditions.py --map greek_island       # one map, all conditions
  python scripts/run_all_conditions.py --conditions 0,3         # all maps, conditions 0 and 3
  python scripts/run_all_conditions.py --map greek_island --conditions 0,3

Usage (remote, orchestrator on one machine, simulator on another):
  # On the simulator machine:
  python scripts/run_sim_controller.py --port 9002

  # On the orchestrator machine:
  python scripts/run_all_conditions.py --sim-controller 192.168.0.101:9002 --sim_host 192.168.0.101
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rvln.config import (
    DEFAULT_DIARY_MODE,
    DEFAULT_LLM_MODEL,
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
    discover_tasks,
    sanitize_run_label,
)
from rvln.maps import SUPPORTED_MAPS
from rvln.paths import BATCH_SCRIPT, REPO_ROOT, UAV_FLOW_EVAL
from requests.exceptions import ReadTimeout, ConnectionError as RequestsConnectionError
from rvln.sim.env_setup import (
    import_batch_module,
    load_env_vars,
    setup_sim_env,
)

logger = logging.getLogger(__name__)

ALL_CONDITIONS = list(range(7))

CONDITION_MODULES = {
    0: ("run_integration", "run_integrated_control_loop"),
    1: ("run_condition1_naive", "run_naive_control_loop"),
    2: ("run_condition2_llm_planner", "run_llm_planner_control_loop"),
    3: ("run_condition3_open_loop", "run_open_loop_control_loop"),
    4: ("run_condition4_single_frame", "run_single_frame_control_loop"),
    5: ("run_condition5_grid_only", "run_grid_only_control_loop"),
    6: ("run_condition6_text_only", "run_text_only_control_loop"),
}

CONDITION_PREFIXES = {
    0: "c0_full_system",
    1: "c1_naive",
    2: "c2_llm_planner",
    3: "c3_open_loop",
    4: "c4_single_frame",
    5: "c5_grid_only",
    6: "c6_text_only",
}


_SIMULATOR_SCRIPT = Path(__file__).resolve().parent / "run_simulator.py"


class SimManager:
    """Manages the simulator lifecycle (local subprocess or remote controller)."""

    def __init__(self, controller_url: Optional[str] = None):
        self._controller_url = controller_url
        self._local_proc: Optional[subprocess.Popen] = None

    @property
    def is_remote(self) -> bool:
        return self._controller_url is not None

    def start(
        self,
        map_info,
        sim_port: int,
        api_port: int,
        time_dilation: int,
        seed: int,
        startup_timeout: float = 120.0,
    ) -> None:
        if self.is_remote:
            self._start_remote(map_info, sim_port, api_port, time_dilation, seed, startup_timeout)
        else:
            self._start_local(map_info, sim_port, api_port, time_dilation, seed, startup_timeout)

    def stop(self) -> None:
        if self.is_remote:
            self._stop_remote()
        else:
            self._stop_local()

    def _start_remote(self, map_info, sim_port, api_port, time_dilation, seed, timeout):
        import requests

        url = f"{self._controller_url}/start"
        payload = {
            "scene": map_info.name,
            "port": sim_port,
            "api_port": api_port,
            "time_dilation": time_dilation,
            "seed": seed,
            "startup_timeout": timeout,
        }
        logger.info("Requesting remote simulator start: scene=%s", map_info.name)
        resp = requests.post(url, json=payload, timeout=timeout + 30)
        if not resp.ok:
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            raise RuntimeError(
                f"Remote controller failed to start simulator for '{map_info.name}': "
                f"{body.get('error', resp.text)}"
            )
        result = resp.json()
        logger.info("Remote simulator ready: %s", result)

    def _stop_remote(self):
        import requests

        url = f"{self._controller_url}/stop"
        logger.info("Requesting remote simulator stop...")
        try:
            resp = requests.post(url, timeout=30)
            if resp.ok:
                logger.info("Remote simulator stopped: %s", resp.json())
            else:
                logger.warning("Remote stop returned %d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.warning("Remote stop request failed: %s", e)

    def _start_local(self, map_info, sim_port, api_port, time_dilation, seed, startup_timeout):
        cmd = [
            sys.executable, str(_SIMULATOR_SCRIPT),
            "--scene", map_info.name,
            "--port", str(sim_port),
            "--api-port", str(api_port),
            "--time_dilation", str(time_dilation),
            "--seed", str(seed),
        ]
        logger.info("Starting simulator locally: %s", " ".join(cmd))
        self._local_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if not self._wait_for_api("127.0.0.1", api_port, timeout=startup_timeout):
            self._local_proc.terminate()
            self._local_proc.wait(timeout=10)
            self._local_proc = None
            raise RuntimeError(
                f"Simulator for {map_info.name} did not become ready within {startup_timeout}s"
            )
        logger.info("Local simulator ready: map=%s, api_port=%d", map_info.name, api_port)

    def _stop_local(self, timeout: float = 15.0):
        proc = self._local_proc
        if proc is None or proc.poll() is not None:
            self._local_proc = None
            return
        pgid = proc.pid  # proc is session leader (start_new_session=True)
        logger.info("Stopping local simulator (PGID %d)...", pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            self._local_proc = None
            return
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Simulator did not exit in %ds, sending SIGKILL", timeout)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
        self._local_proc = None
        logger.info("Simulator stopped.")

    @staticmethod
    def _wait_for_api(host: str, port: int, timeout: float = 120.0, interval: float = 3.0) -> bool:
        import requests

        url = f"http://{host}:{port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(url, timeout=5)
                if resp.ok and resp.json().get("initialized"):
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False


def _import_control_loop(condition: int):
    """Import and return the control loop function for a condition."""
    module_name, func_name = CONDITION_MODULES[condition]
    mod = __import__(module_name)
    return getattr(mod, func_name)


def _restart_sim(sim_manager: SimManager, map_info, args, batch):
    """Restart the simulator and return a fresh env connection."""
    logger.info("  Restarting simulator to recover from timeout...")
    sim_manager.stop()
    time.sleep(5)
    sim_manager.start(
        map_info,
        sim_port=args.sim_port,
        api_port=args.sim_api_port,
        time_dilation=args.time_dilation,
        seed=args.seed,
        startup_timeout=args.startup_timeout,
    )
    env = setup_sim_env(
        int(args.time_dilation), int(args.seed), batch,
        sim_host=args.sim_host, sim_api_port=args.sim_api_port,
    )
    logger.info("  Simulator restarted successfully.")
    return env


MAX_CONSECUTIVE_TIMEOUTS = 2


def _run_condition_tasks(
    condition: int,
    tasks: List[Dict[str, Any]],
    env,
    batch,
    server_url: str,
    map_info,
    args,
    sim_manager: SimManager,
):
    control_loop = _import_control_loop(condition)
    results_dir = Path(args.results_dir) / f"condition{condition}" / map_info.task_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = CONDITION_PREFIXES[condition]
    drone_cam_id = env.drone_cam_id
    consecutive_timeouts = 0

    completed_ids = get_completed_task_ids(results_dir)
    if completed_ids:
        logger.info("  Found %d completed task(s) in %s", len(completed_ids), results_dir)

    idx = 0
    while idx < len(tasks):
        task = tasks[idx]
        task_id = task.get("task_id", "")
        if task_id and task_id in completed_ids:
            logger.info("  Skipping already completed task: %s", task_id)
            idx += 1
            continue

        logger.info(
            "  Task %d/%d: '%s'",
            idx + 1, len(tasks), task["instruction"][:80],
        )

        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        task_label = task.get("task_id") or sanitize_run_label(task["instruction"], max_len=30)
        run_name = f"{prefix}__{task_label}__{ts}"
        run_dir = results_dir / run_name

        common_kwargs = dict(
            env=env,
            batch=batch,
            task=task,
            server_url=server_url,
            run_dir=run_dir,
            drone_cam_id=drone_cam_id,
            save_mp4=args.save_mp4,
            mp4_fps=args.mp4_fps,
            seed=args.seed,
            time_dilation=args.time_dilation,
            env_id=map_info.env_id,
        )

        try:
            if condition == 0:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    diary_mode=args.diary_mode,
                )
            elif condition == 1:
                run_info = control_loop(**common_kwargs)
            elif condition == 2:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    diary_mode=args.diary_mode,
                )
            elif condition == 3:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    converter_model=args.llm_model,
                )
            elif condition == 4:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                )
            elif condition == 5:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    monitor_model=args.monitor_model,
                    diary_mode=args.diary_mode,
                )
            elif condition == 6:
                run_info = control_loop(
                    **common_kwargs,
                    llm_model=args.llm_model,
                    vlm_model=args.vlm_model,
                )

            consecutive_timeouts = 0
            logger.info(
                "  Run saved to %s (steps=%s, stop=%s)",
                run_dir,
                run_info.get("total_steps", "?"),
                run_info.get("stop_reason", "?"),
            )
        except KeyboardInterrupt:
            logger.info("  Task interrupted.")
            raise
        except (ReadTimeout, RequestsConnectionError) as e:
            consecutive_timeouts += 1
            logger.error("  Task failed (timeout %d/%d): %s",
                         consecutive_timeouts, MAX_CONSECUTIVE_TIMEOUTS, e)
            if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                logger.warning("  %d consecutive timeouts, restarting simulator...",
                               consecutive_timeouts)
                try:
                    env = _restart_sim(sim_manager, map_info, args, batch)
                    drone_cam_id = env.drone_cam_id
                    consecutive_timeouts = 0
                    continue  # retry same task without advancing idx
                except Exception as restart_err:
                    logger.error("  Simulator restart failed: %s", restart_err)
                    raise
        except Exception as e:
            consecutive_timeouts = 0
            logger.error("  Task failed: %s", e, exc_info=True)

        idx += 1


def _resolve_maps(map_arg: Optional[str]) -> List:
    """Resolve which maps to run. None means all maps."""
    if map_arg is None:
        return list(SUPPORTED_MAPS.values())
    matching = None
    for m in SUPPORTED_MAPS.values():
        if m.task_dir_name == map_arg:
            matching = m
            break
    if matching is None:
        valid = ", ".join(m.task_dir_name for m in SUPPORTED_MAPS.values())
        raise SystemExit(f"Unknown map '{map_arg}'. Valid task_dir_names: {valid}")
    return [matching]


def _run_map(
    map_info,
    conditions: List[int],
    args,
    batch,
    server_url: str,
    env,
    sim_manager: SimManager,
):
    """Run all conditions for a single map on an already-connected env."""
    tasks = discover_tasks(REPO_ROOT / "tasks" / map_info.task_dir_name)
    if not tasks:
        logger.warning("No tasks found in tasks/%s/, skipping map.", map_info.task_dir_name)
        return
    logger.info("Found %d shared task(s) for map '%s'", len(tasks), map_info.task_dir_name)

    for cond in conditions:
        if cond not in CONDITION_MODULES:
            logger.warning("Unknown condition %d, skipping", cond)
            continue

        logger.info(
            "\n===== Condition %d (%s): %d task(s) =====",
            cond, CONDITION_PREFIXES[cond], len(tasks),
        )

        _run_condition_tasks(
            condition=cond,
            tasks=tasks,
            env=env,
            batch=batch,
            server_url=server_url,
            map_info=map_info,
            args=args,
            sim_manager=sim_manager,
        )

        logger.info("===== Condition %d finished =====\n", cond)


def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Run all conditions back-to-back, optionally across multiple maps",
    )
    parser.add_argument("--map", type=str, default=None,
                        help="Map task_dir_name (e.g. greek_island). Omit to run all maps.")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated condition numbers to run (default: all 0-6)")
    parser.add_argument("-t", "--time_dilation", type=int, default=DEFAULT_TIME_DILATION)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-p", "--server_port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--server_host", type=str, default=DEFAULT_SERVER_HOST)
    parser.add_argument("--sim_host", type=str, default=DEFAULT_SIM_HOST)
    parser.add_argument("--sim_port", type=int, default=DEFAULT_SIM_PORT)
    parser.add_argument("--sim_api_port", type=int, default=DEFAULT_SIM_API_PORT)
    parser.add_argument("--sim-controller", type=str, default=None,
                        help="Remote sim controller address (host:port). "
                             "Omit to manage the simulator locally as a subprocess.")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--monitor_model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--diary_mode", type=str, default=DEFAULT_DIARY_MODE)
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--mp4-fps", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0,
                        help="Seconds to wait for simulator startup")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    if args.conditions:
        conditions = [int(c.strip()) for c in args.conditions.split(",")]
    else:
        conditions = ALL_CONDITIONS

    maps_to_run = _resolve_maps(args.map)

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    batch = import_batch_module()
    os.chdir(str(UAV_FLOW_EVAL))

    server_url = f"http://{args.server_host}:{args.server_port}/predict"

    controller_url = None
    if args.sim_controller:
        host_port = args.sim_controller
        if not host_port.startswith("http"):
            host_port = f"http://{host_port}"
        controller_url = host_port
    sim_manager = SimManager(controller_url=controller_url)

    try:
        for map_idx, target_map in enumerate(maps_to_run):
            logger.info(
                "\n########## Map %d/%d: %s ##########",
                map_idx + 1, len(maps_to_run), target_map.name,
            )

            sim_manager.start(
                target_map,
                sim_port=args.sim_port,
                api_port=args.sim_api_port,
                time_dilation=args.time_dilation,
                seed=args.seed,
                startup_timeout=args.startup_timeout,
            )

            try:
                env = setup_sim_env(
                    int(args.time_dilation), int(args.seed), batch,
                    sim_host=args.sim_host, sim_api_port=args.sim_api_port,
                )
                map_info = env.get_map_info()

                if map_info.task_dir_name != target_map.task_dir_name:
                    raise RuntimeError(
                        f"Map mismatch: expected '{target_map.name}' but simulator "
                        f"is running '{map_info.name}'"
                    )

                logger.info("Connected to simulator: map=%s", map_info.name)
                logger.info("Conditions to run: %s", conditions)

                _run_map(map_info, conditions, args, batch, server_url, env, sim_manager)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(
                    "Map '%s' aborted, moving to next map: %s", target_map.name, e,
                )
            finally:
                sim_manager.stop()

            if map_idx < len(maps_to_run) - 1:
                logger.info("Waiting 5s before starting next map...")
                time.sleep(5)

    except KeyboardInterrupt:
        logger.info("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()
