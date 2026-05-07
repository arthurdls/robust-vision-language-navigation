#!/usr/bin/env python3
"""
Run the MiniNav real-drone integration pipeline.

Drives a real (or mocked) drone via the boieng wire format using the OpenVLA
server, LTL planner, and live goal adherence monitor. Results land under
results/hardware/run_<timestamp>/. See README "Running on Hardware (MiniNav)"
for the full terminal layout.

Wire format (matches boieng_mininav.py): each packet is 5 float32 values,
[frame_count, vx, vy, vz, yaw_rate], where vx/vy/vz are in m/s and
yaw_rate is in rad/s. Internally the pipeline keeps OpenVLA's cm
emission convention end to end (per-step delta, magnitude clip in
cm/s, dead-reckoning); the wire boundary multiplies by
scale_output_translation (default 0.01 -> m/s) before sending.

Configure runs by editing the CONFIG block below. Hardware always runs in
time-based monitor mode; frame-mode options are intentionally absent.
Anything passed on the command line still wins over CONFIG (handy for
one-off overrides without editing the file). Every value below is a
literal default copied from the upstream argparse so what you see is
exactly what you get.
"""

import sys
from pathlib import Path

# Computed once so the visible default path is the actual absolute one
# the run will use. Resolves to <repo_root>/results/hardware regardless
# of where you cd before invoking the script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RESULTS_DIR = str(_REPO_ROOT / "results" / "hardware")

# ============================================================================
# CONFIG -- edit these for your run, then `python scripts/run_hardware.py`.
# Every key is a literal value (no None, no DEFAULT_X references). Empty
# strings ("") mark genuinely-optional alternatives -- they're translated
# to "not passed" so argparse's None default takes effect; e.g. an empty
# instruction triggers an interactive stdin prompt.
# ============================================================================

CONFIG = {
    # ---- Task ------------------------------------------------------------
    # Mission instruction. "" => the run prompts on stdin at startup.
    # Set a string here to skip the prompt.
    "instruction": "",
    # Starting world pose: "x,y,z,yaw_deg".
    "initial_position": "0,0,0,0",
    # Run-output directory. Defaults to <repo_root>/results/hardware/.
    "results_dir": _DEFAULT_RESULTS_DIR,

    # ---- Camera (set exactly one source) ---------------------------------
    # Local cv2 device index (USB / V4L2). 4 = MiniNav USB cam on Jetson.
    "camera": 0,
    # HTTP frame source, e.g. "http://127.0.0.1:8081/frame" for the mock.
    # Set this XOR camera_pipeline to override the cv2 capture.
    "camera_url": "",
    # GStreamer pipeline (Jetson MIPI-CSI). Must end in 'appsink'.
    "camera_pipeline": "",
    "fps": 30,
    "camera_retries": 15,
    "camera_init_timeout": 8.0,

    # ---- Recording -------------------------------------------------------
    # Persist per-step PNGs under run_dir/frames + recording_log.jsonl
    # alongside the always-on playback.mp4 video.
    "record": True,
    "record_fps": 30.0,

    # ---- Control server (boieng wire) ------------------------------------
    "control_host": "192.168.0.101",
    "control_port": 8080,
    "control_retries": 10,
    "control_retry_sleep": 2.0,
    "command_dt_s": 0.1,
    # Wire-output scaling: multipliers applied to vx/vy/vz and yaw_rate
    # at the wire boundary. Internal pipeline runs in cm/s + rad/s;
    # defaults convert vx/vy/vz to m/s and leave yaw_rate as rad/s.
    "scale_output_translation": 0.01,
    "scale_output_rotation": -3.0,
    # Per-step magnitude clip applied BEFORE wire scaling. User-facing
    # units: meters/second for translation, degrees/second for rotation.
    # Defaults give 0.5 m/s + 20 deg/s on the wire with the default
    # output scales. Translation is clipped on the 3D vector norm so
    # heading is preserved; yaw is sign-preserved.
    "max_translation_m_s": 0.6,
    "max_rotation_deg_s": 20.0,

    # ---- OpenVLA ---------------------------------------------------------
    "openvla_predict_url": "http://127.0.0.1:5007/predict",

    # ---- Small-motion auto-converge --------------------------------------
    # When OpenVLA emits N consecutive "small" steps the run forces a
    # convergence VLM call (drone is presumed to have stopped moving).
    # action_small_delta_pos is per-axis cm; action_small_delta_yaw is
    # degrees; action_small_steps is the consecutive-step count.
    "action_small_delta_pos": 3.0,
    "action_small_delta_yaw": 1.0,
    "action_small_steps": 1000,

    # ---- LTL planner / subgoal converter ---------------------------------
    # gpt-4o (cheaper, structured text-to-text). Mirrors DEFAULT_LLM_MODEL
    # in src/rvln/config.py.
    "llm_model": "gpt-4o",

    # ---- Goal-adherence monitor (TIME MODE ONLY) -------------------------
    # gpt-5.4 for the visual monitor. Mirrors DEFAULT_VLM_MODEL.
    "monitor_model": "gpt-5.4",
    "diary_check_interval_s": 3.0,
    # Spacing (seconds) between cells in the 9-frame global VLM grid.
    # Independent from diary_check_interval_s -- you can sample the
    # diary every 1 s while the global grid still steps every 3 s, and
    # the grid will only shift by one cell when a new spacing boundary
    # is crossed (consistent past frames across consecutive
    # checkpoints). "" -> inherit diary_check_interval_s.
    "global_grid_spacing_s": 6.0,
    # Spacing (seconds) between the prev and curr frames in the 2-frame
    # local "what changed" VLM grid. Without this the local prompt
    # compared two ~100 ms-apart frames (essentially identical) and the
    # VLM had nothing to describe. "" cascades to global_grid_spacing_s,
    # which itself cascades to diary_check_interval_s.
    "local_grid_spacing_s": "",
    "max_steps_per_subgoal": 500,
    "max_seconds_per_subgoal": 120.0,
    "max_corrections": 20,
    "stall_window": 5,
    "stall_threshold": 0.05,
    "stall_completion_floor": 0.5,

    # ---- Pipelined-monitor buffer (TIME MODE ONLY) -----------------------
    # In-flight cap on concurrent checkpoint VLM calls. With 1 s dispatch
    # and ~3 s VLM latency only ~3 calls are in flight; this cap mainly
    # protects against a hung OpenAI endpoint piling up workers.
    "monitor_max_inflight": 16,
    # Per-step timeout for the reorder buffer. If a checkpoint's local or
    # global call has not returned this long after dispatch, its slot is
    # skipped so the dashboard does not stall.
    "monitor_dispatch_timeout_s": 30.0,

    # ---- Pose source (exactly one of these) -----------------------------
    # dead_reckoning ON by default since most hardware runs do not have a
    # live odometry feed wired up. To switch to real odometry, set
    # odom_http_url (or odom_udp_port > 0) AND flip dead_reckoning to
    # False -- they are mutually exclusive and the run aborts if both
    # are configured.
    "odom_http_url": "",
    "odom_udp_host": "0.0.0.0",
    "odom_udp_port": 0,
    "odom_stale_timeout_s": 1.0,
    "odom_poll_hz": 50.0,
    "dead_reckoning": True,

    # ---- Misc -----------------------------------------------------------
    # Optional .env overlay layered on top of .env / .env.local. "" =
    # no overlay.
    "extra_env_file": "",
    "log_level": "INFO",
    # Shortcut: True == --log_level DEBUG. Same effect as passing -v on
    # the CLI. Suppresses third-party HTTP / image-encoding debug noise.
    "verbose": False,
    # Live monitor dashboard (Tkinter popup) showing the last
    # checkpoint's local + global prompt images, the diary, the last
    # global response, and the last convergence. Set to False for
    # headless / SSH runs without a display.
    "dashboard": True,
}

# argparse uses dashes for these flags but Python identifiers can't have
# dashes, so we keep underscore keys in CONFIG and translate at flag-build
# time.
_DASHED_FLAGS = {
    "dead_reckoning": "dead-reckoning",
    "extra_env_file": "extra-env-file",
}

# CONFIG keys whose argparse counterpart is a "store_false" / "disable"
# flag with default True. CONFIG reads naturally as "enabled by default";
# we translate False -> "--<flag>" and True -> "(nothing, rely on default)".
_INVERTED_BOOL_FLAGS = {
    "dashboard": "no-dashboard",
}


def _build_argv(cfg: dict) -> list[str]:
    """Translate CONFIG into argv flags for rvln.mininav.interface.main.

    Hardware always runs time-based monitoring, so --diary-mode is forced
    to 'time' here; remove it from CONFIG to avoid surprising overrides.
    Empty strings are skipped so the genuinely-optional alternatives
    (instruction, camera_url, camera_pipeline, odom_http_url,
    extra_env_file) fall through to argparse's None default and the
    consuming code's "if not value" branch.
    """
    argv: list[str] = ["--diary-mode", "time"]
    for key, value in cfg.items():
        if value == "":
            continue
        if key in _INVERTED_BOOL_FLAGS:
            # Default-true flag whose CLI form turns it OFF. Skip when
            # True; pass --no-X when False.
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


# src/ layout: allow `python scripts/run_hardware.py` without `pip install -e .`
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.mininav.interface import main


# Appended to the goal-adherence monitor prompts at hardware startup so the
# VLM stays patient with real-drone rotation latency. Sim eval imports the
# unpatched templates from rvln.ai.prompts and is unaffected.
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


# Sim eval's convergence prompts tell the VLM to "default to turning RIGHT to
# search" when no directional evidence is available. On hardware that bias
# fights against operator-requested turn directions, so we strip the whole
# "When the target cannot be located ... new ground each correction." chunk
# out of the DIARY and TEXT_ONLY convergence templates. The substrings below
# match the corresponding source text in rvln.ai.prompts verbatim; if those
# templates are reworded the assertion in _apply_hardware_prompt_patches
# will catch the drift.
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
    """Patch the goal-adherence monitor's prompt templates for hardware:

    1. Append the HARDWARE PATIENCE addendum to the four global / convergence
       templates actually used at runtime (DIARY_* and TEXT_ONLY_*).
    2. Strip the "default to turning RIGHT to search" guidance out of the
       two convergence templates so the VLM does not bias against an
       operator-requested turn direction on real hardware.

    GRID_ONLY_* / SINGLE_FRAME_* templates are sim-eval-only and untouched.
    """
    import rvln.ai.goal_adherence_monitor as gam

    if _DIARY_DEFAULT_RIGHT_BLOCK not in gam.CONVERGENCE_PROMPT_TEMPLATE:
        raise RuntimeError(
            "DIARY_CONVERGENCE_PROMPT no longer contains the expected "
            "'default to turning RIGHT' block; update "
            "_DIARY_DEFAULT_RIGHT_BLOCK in run_hardware.py to match."
        )
    if _TEXT_ONLY_DEFAULT_RIGHT_BLOCK not in gam.TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE:
        raise RuntimeError(
            "TEXT_ONLY_CONVERGENCE_PROMPT no longer contains the expected "
            "'default to turning RIGHT' block; update "
            "_TEXT_ONLY_DEFAULT_RIGHT_BLOCK in run_hardware.py to match."
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
    _apply_hardware_prompt_patches()
    # CONFIG first so CLI overrides win (argparse takes the last value for
    # repeated options). --help still works because it's a CLI arg.
    sys.argv = [sys.argv[0]] + _build_argv(CONFIG) + sys.argv[1:]
    main()
