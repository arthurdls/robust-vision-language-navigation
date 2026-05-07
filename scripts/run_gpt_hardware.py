#!/usr/bin/env python3
"""
Run the MiniNav real-drone integration pipeline with GPT-5.4 driving.

Same as ``run_hardware.py`` but the OpenVLA model is surgically removed.
Instead, the goal-adherence monitor (gpt-5.4) emits a discrete
``drive_action`` on every checkpoint -- one of
``{"turn_left", "turn_right", "move_forward", "stop"}`` -- and the run
loop emits the corresponding constant-rate velocity at the existing
10 Hz wire cadence:

  - move_forward : vx = GPT_DRIVE_FORWARD_M_S * 100 cm/s, others zero
  - turn_left    : vyaw = -radians(GPT_DRIVE_TURN_DEG_S)
  - turn_right   : vyaw = +radians(GPT_DRIVE_TURN_DEG_S)
  - stop         : zeros

Between monitor checkpoints (~diary_check_interval_s seconds) the loop
re-emits the same drive_action; when a fresh checkpoint result lands,
the action flips and the loop continues unchanged.

You do NOT need to run the OpenVLA server for this script. The HTTP
client is replaced by a stub that synthesises action_poses from the
latest GPT-issued drive_action; ``openvla_predict_url`` is never hit.

All overrides are monkey-patches installed at startup. No edits to
``goal_adherence_monitor.py`` / ``interface.py`` / ``prompts.py`` are
required: this script flips the behavior at runtime.
"""

import math
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Computed once so the visible default path is the actual absolute one
# the run will use. Resolves to <repo_root>/results/hardware regardless
# of where you cd before invoking the script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS_DIR = str(_REPO_ROOT / "results" / "hardware_gpt")

# ============================================================================
# GPT DRIVE TUNING -- the rate and frequency knobs for the GPT-driven loop.
# Tweak these to change how aggressively the drone moves on each GPT-issued
# action. Both default to the safety caps in CONFIG (max_translation_m_s and
# max_rotation_deg_s), so by default the wire view is exactly 0.6 m/s
# forward and 20 deg/s turn whenever GPT picks the matching action. Set
# them lower for more cautious flight; the safety caps below still apply
# as a hard ceiling.
# ============================================================================

GPT_DRIVE_FORWARD_M_S = 0.6   # forward velocity for the "move_forward" action
GPT_DRIVE_TURN_DEG_S  = 20.0  # yaw rate for "turn_left" / "turn_right"

# ============================================================================
# CONFIG -- edit these for your run, then `python scripts/run_gpt_hardware.py`.
# Same shape as run_hardware.py's CONFIG. OpenVLA-specific knobs are still
# present (the runner threads them through), but the OpenVLA HTTP client is
# stubbed out at startup so ``openvla_predict_url`` never gets hit and the
# OpenVLA server does NOT need to be running.
# ============================================================================

CONFIG = {
    # ---- Task ------------------------------------------------------------
    "instruction": "",
    "initial_position": "0,0,0,0",
    "results_dir": _DEFAULT_RESULTS_DIR,

    # ---- Camera (set exactly one source) ---------------------------------
    "camera": 0,
    "camera_url": "",
    "camera_pipeline": "",
    "fps": 30,
    "camera_retries": 15,
    "camera_init_timeout": 8.0,

    # ---- Recording -------------------------------------------------------
    "record": True,
    "record_fps": 30.0,

    # ---- Control server (boieng wire) ------------------------------------
    "control_host": "192.168.0.101",
    "control_port": 8080,
    "control_retries": 10,
    "control_retry_sleep": 2.0,
    "command_dt_s": 0.1,
    "scale_output_translation": 0.01,
    "scale_output_rotation": -3.0,
    # Per-step magnitude clip applied BEFORE wire scaling. With GPT driving,
    # these caps double as the constant rates -- GPT_DRIVE_FORWARD_M_S /
    # GPT_DRIVE_TURN_DEG_S below default to these values so the wire sees
    # exactly 0.6 m/s forward and 20 deg/s turn.
    "max_translation_m_s": 0.6,
    "max_rotation_deg_s": 20.0,

    # ---- OpenVLA (stubbed out -- URL is never hit) -----------------------
    "openvla_predict_url": "http://127.0.0.1:5007/predict",
    "openvla_dead_zone": True,

    # ---- Small-motion auto-converge (effectively disabled) ---------------
    # GPT issues "stop" directly when it wants the drone to hold; we leave
    # the auto-converge thresholds as in run_hardware (steps=1000) so the
    # heuristic effectively never triggers.
    "action_small_delta_pos": 3.0,
    "action_small_delta_yaw": 1.0,
    "action_small_steps": 1000,

    # ---- LTL planner / subgoal converter ---------------------------------
    "llm_model": "gpt-4o",

    # ---- Goal-adherence monitor (TIME MODE ONLY) -------------------------
    "monitor_model": "gpt-5.4",
    "diary_check_interval_s": 3.0,
    "global_grid_spacing_s": 6.0,
    "local_grid_spacing_s": "",
    "max_steps_per_subgoal": 500,
    "max_seconds_per_subgoal": 120.0,
    "max_corrections": 20,
    "stall_window": 5,
    "stall_threshold": 0.05,
    "stall_completion_floor": 0.5,

    # ---- Pipelined-monitor buffer (TIME MODE ONLY) -----------------------
    "monitor_max_inflight": 16,
    "monitor_dispatch_timeout_s": 30.0,

    # ---- Pose source (exactly one of these) -----------------------------
    "odom_http_url": "",
    "odom_udp_host": "0.0.0.0",
    "odom_udp_port": 0,
    "odom_stale_timeout_s": 1.0,
    "odom_poll_hz": 50.0,
    "dead_reckoning": True,

    # ---- Misc -----------------------------------------------------------
    "extra_env_file": "",
    "log_level": "INFO",
    "verbose": False,
    "dashboard": True,
}

_VALID_DRIVE_ACTIONS = {"turn_left", "turn_right", "move_forward", "stop"}

# argparse uses dashes for these flags but Python identifiers can't have
# dashes, so we keep underscore keys in CONFIG and translate at flag-build
# time.
_DASHED_FLAGS = {
    "dead_reckoning": "dead-reckoning",
    "extra_env_file": "extra-env-file",
}

_INVERTED_BOOL_FLAGS = {
    "dashboard": "no-dashboard",
    "openvla_dead_zone": "no-openvla-dead-zone",
}


def _build_argv(cfg: dict) -> list[str]:
    """Translate CONFIG into argv flags for rvln.mininav.interface.main."""
    argv: list[str] = ["--diary-mode", "time"]
    for key, value in cfg.items():
        if value == "":
            continue
        if key in _INVERTED_BOOL_FLAGS:
            if value is False:
                argv.append("--" + _INVERTED_BOOL_FLAGS[key])
            continue
        flag = "--" + _DASHED_FLAGS.get(key, key)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv


# src/ layout: allow `python scripts/run_gpt_hardware.py` without `pip install -e .`
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ============================================================================
# GPT drive state -- the bridge between gpt-5.4's checkpoint output and the
# wire-level velocity loop. The monitor parser side-effects updates here;
# the patched to_command_from_action_pose reads from here.
# ============================================================================


class _GPTDriveState:
    """Process-wide latest GPT-issued drive action.

    Single-writer (the monitor's parse path) / single-reader (the per-step
    wire loop) in practice, but the lock keeps it safe regardless.
    """

    def __init__(self) -> None:
        self._action: str = "stop"
        self._lock = threading.Lock()

    @property
    def action(self) -> str:
        with self._lock:
            return self._action

    def set(self, action: str) -> None:
        if action not in _VALID_DRIVE_ACTIONS:
            return
        with self._lock:
            self._action = action

    def reset(self) -> None:
        with self._lock:
            self._action = "stop"


_DRIVE_STATE = _GPTDriveState()


# ============================================================================
# Velocity synthesis. The patched to_command function ignores the OpenVLA
# action_pose entirely and returns a constant velocity sized to match what
# the (now-removed) OpenVLA full-speed output would have produced.
# ============================================================================


def _drive_action_to_velocity(action: str) -> np.ndarray:
    """Map a drive_action label to an internal velocity vector.

    Internal pipeline units: cm/s for vx/vy/vz, rad/s for vyaw. The wire
    boundary multiplies by scale_output_translation (default 0.01 -> m/s)
    and scale_output_rotation (default -3.0).
    """
    if action == "move_forward":
        vx = GPT_DRIVE_FORWARD_M_S * 100.0  # m/s -> cm/s
        return np.array([vx, 0.0, 0.0, 0.0], dtype=np.float32)
    if action == "turn_left":
        vyaw = -math.radians(GPT_DRIVE_TURN_DEG_S)
        return np.array([0.0, 0.0, 0.0, vyaw], dtype=np.float32)
    if action == "turn_right":
        vyaw = +math.radians(GPT_DRIVE_TURN_DEG_S)
        return np.array([0.0, 0.0, 0.0, vyaw], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ============================================================================
# Monkey-patches. Installed once at startup, before main() touches anything
# that would resolve OpenVLAClient or to_command_from_action_pose.
# ============================================================================


_DRIVE_ACTION_ADDENDUM = (
    "\n\nDRIVE ACTION REQUIRED -- this run has the OpenVLA model removed; "
    "you (the monitor) are the sole controller. In addition to every field "
    "in the JSON object above, your output JSON MUST include a "
    "\"drive_action\" key whose value is exactly one of "
    "\"turn_left\", \"turn_right\", \"move_forward\", or \"stop\". The drone "
    "will execute that action at constant rate "
    f"({GPT_DRIVE_FORWARD_M_S:.2f} m/s forward, "
    f"{GPT_DRIVE_TURN_DEG_S:.1f} deg/s turn) continuously until your next "
    "checkpoint (~3 seconds later). Pick the single action that, applied "
    "for the next ~3 seconds, would best advance the active subgoal given "
    "the most recent visual evidence. Use \"stop\" when the subgoal is "
    "complete, when the drone should hold position, or when you are "
    "uncertain which way to go. Never omit drive_action."
)


def _install_gpt_drive_overrides() -> None:
    """Wire all the runtime overrides that replace OpenVLA with GPT control."""
    import rvln.ai.goal_adherence_monitor as gam
    import rvln.mininav.interface as iface

    # 1. Append the drive_action directive to every monitor prompt template
    #    that gpt-5.4 might respond to. These are module-level names re-
    #    bound here; the monitor reads them via the bare name at format
    #    time, so subsequent .format() calls pick up the patched value.
    gam.GLOBAL_PROMPT_TEMPLATE += _DRIVE_ACTION_ADDENDUM
    gam.CONVERGENCE_PROMPT_TEMPLATE += _DRIVE_ACTION_ADDENDUM
    gam.TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE += _DRIVE_ACTION_ADDENDUM
    gam.TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE += _DRIVE_ACTION_ADDENDUM

    # 2. Wrap the JSON parser so every successful parse with a drive_action
    #    key updates the shared state. Defensive: if the parsed dict says
    #    complete=true OR should_stop=true, force drive_action=stop so a
    #    "subgoal complete" verdict cannot leave the drone marching forward
    #    until the next subgoal's first monitor result.
    _orig_parse = gam.GoalAdherenceMonitor._parse_json_response

    def _patched_parse(response: str) -> Optional[Dict[str, Any]]:
        parsed = _orig_parse(response)
        if isinstance(parsed, dict):
            raw = parsed.get("drive_action")
            if isinstance(raw, str) and raw in _VALID_DRIVE_ACTIONS:
                _DRIVE_STATE.set(raw)
            # Safety override: stop on completion / stop verdicts.
            complete = parsed.get("complete")
            should_stop = parsed.get("should_stop")
            if complete is True or should_stop is True:
                _DRIVE_STATE.set("stop")
        return parsed

    # _parse_json_response is a staticmethod; rebind as staticmethod so
    # the unbound-call sites (`self._parse_json_response(...)`) still work.
    gam.GoalAdherenceMonitor._parse_json_response = staticmethod(_patched_parse)

    # 3. Replace OpenVLAClient with a stub. predict() returns a dummy
    #    1-element action_poses list that the patched to_command function
    #    will ignore. reset_model() resets the shared state so each new
    #    subgoal starts in "stop" until GPT picks something.
    class _StubOpenVLAClient:
        def __init__(self, predict_url: str = "", timeout_s: float = 30.0):
            self.predict_url = predict_url
            self.timeout_s = timeout_s

        @property
        def reset_url(self) -> str:
            return self.predict_url

        def reset_model(self) -> None:
            _DRIVE_STATE.reset()

        def predict(self, image_bgr: Any, proprio: Any, instr: str) -> Dict[str, Any]:
            # Single zero-pose entry; to_command_from_action_pose is
            # patched below to ignore action_pose entirely.
            return {"action": [[0.0, 0.0, 0.0, 0.0]]}

    iface.OpenVLAClient = _StubOpenVLAClient

    # 4. Replace to_command_from_action_pose with a state-driven version
    #    that ignores its inputs and returns the constant-rate velocity
    #    for the latest GPT-issued drive_action. Safety caps still apply
    #    via _clip_velocity so a misconfigured GPT_DRIVE_* constant cannot
    #    bypass max_translation_cm_s / max_rotation_rad_s.
    _clip_velocity = iface._clip_velocity

    def _patched_to_command(
        action_pose: List[float],
        current_relative_pose: List[float],
        max_translation_cm_s: float,
        max_rotation_rad_s: float,
        apply_dead_zone: bool = True,
    ) -> np.ndarray:
        v = _drive_action_to_velocity(_DRIVE_STATE.action)
        vx, vy, vz, vyaw = _clip_velocity(
            float(v[0]), float(v[1]), float(v[2]), float(v[3]),
            max_translation_cm_s, max_rotation_rad_s,
        )
        return np.array([vx, vy, vz, vyaw], dtype=np.float32)

    iface.to_command_from_action_pose = _patched_to_command


# ============================================================================
# Hardware patience prompt patches -- copied verbatim from run_hardware.py
# so the existing hardware-only addenda still apply on top of the
# drive_action directive.
# ============================================================================


_HARDWARE_PATIENCE_ADDENDUM = (
    "\n\nHARDWARE PATIENCE -- this run is on real drone hardware: if the "
    "active subgoal or most recent corrective requested a turn in a "
    "specific direction (e.g., \"turn left until you see X\"), do NOT set "
    "should_stop=true and do NOT issue a corrective that reverses the "
    "requested turn direction merely because the rotation is slow or has "
    "not yet produced visual change. Real hardware takes time to execute "
    "a turn; be patient and let the drone finish the requested rotation. "
    "If a different axis is needed, switch to altitude or a forward / "
    "backward move, but never flip the requested turn direction. When no "
    "direction has been requested, pick the side supported by the most "
    "recent evidence (diary, displacement, last known bearing) and stay "
    "with that side across corrections."
)

_DIARY_DEFAULT_RIGHT_BLOCK = """\
toward it. When
  the target cannot be located and there is no directional evidence (diary,
  displacement, or last known bearing) pointing left, default to turning
  RIGHT to search. Always sweeping the same direction prevents the drone
  from oscillating left-right and re-covering the same arc, so it sweeps
  new ground each correction."""

_TEXT_ONLY_DEFAULT_RIGHT_BLOCK = """\
toward it. When
  the target cannot be located and the diary/displacement give no directional
  evidence pointing left, default to turning RIGHT to search. Always sweeping
  the same direction prevents the drone from oscillating left-right and
  re-covering the same arc, so it sweeps new ground each correction."""


def _apply_hardware_prompt_patches() -> None:
    """Same patches run_hardware.py applies. Idempotent w.r.t. drive_action
    addendum because this runs first."""
    import rvln.ai.goal_adherence_monitor as gam

    if _DIARY_DEFAULT_RIGHT_BLOCK not in gam.CONVERGENCE_PROMPT_TEMPLATE:
        raise RuntimeError(
            "DIARY_CONVERGENCE_PROMPT no longer contains the expected "
            "'default to turning RIGHT' block; update "
            "_DIARY_DEFAULT_RIGHT_BLOCK in run_gpt_hardware.py to match."
        )
    if _TEXT_ONLY_DEFAULT_RIGHT_BLOCK not in gam.TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE:
        raise RuntimeError(
            "TEXT_ONLY_CONVERGENCE_PROMPT no longer contains the expected "
            "'default to turning RIGHT' block; update "
            "_TEXT_ONLY_DEFAULT_RIGHT_BLOCK in run_gpt_hardware.py to match."
        )

    gam.CONVERGENCE_PROMPT_TEMPLATE = gam.CONVERGENCE_PROMPT_TEMPLATE.replace(
        _DIARY_DEFAULT_RIGHT_BLOCK, "toward it.",
    )
    gam.TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE = (
        gam.TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE.replace(
            _TEXT_ONLY_DEFAULT_RIGHT_BLOCK, "toward it.",
        )
    )

    gam.GLOBAL_PROMPT_TEMPLATE += _HARDWARE_PATIENCE_ADDENDUM
    gam.CONVERGENCE_PROMPT_TEMPLATE += _HARDWARE_PATIENCE_ADDENDUM
    gam.TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE += _HARDWARE_PATIENCE_ADDENDUM
    gam.TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE += _HARDWARE_PATIENCE_ADDENDUM


if __name__ == "__main__":
    # Order matters: prompt-text patches first so the drive_action addendum
    # sits at the end of the prompt (most recent instruction wins for LLMs).
    _apply_hardware_prompt_patches()
    _install_gpt_drive_overrides()

    # Late import so the OpenVLAClient / to_command_from_action_pose
    # reassignments above are visible inside interface.main(). Imports done
    # before _install_gpt_drive_overrides() would still see the patched
    # names because Python looks up module attributes at call time, but
    # importing here makes the dependency order explicit.
    from rvln.mininav.interface import main

    sys.argv = [sys.argv[0]] + _build_argv(CONFIG) + sys.argv[1:]
    main()
