#!/usr/bin/env python3
"""
Run the MiniNav real-drone integration pipeline.

Drives a real (or mocked) drone via the boieng wire format using the OpenVLA
server, LTL planner, and live goal adherence monitor. Results land under
results/hardware/run_<timestamp>/. See README "Running on Hardware (MiniNav)"
for the full terminal layout.

Wire format (matches boieng_mininav.py): each packet is 5 float32 values,
[frame_count, vx, vy, vz, yaw_rate], where vx/vy/vz are in cm/s and
yaw_rate is in rad/s. Per-step velocities are derived from OpenVLA's
predicted target pose and clipped to <=50 cm/s linear and <=20 deg/s
yaw before being sent.

Configure runs by editing the CONFIG block below. Hardware always runs in
time-based monitor mode; frame-mode options are intentionally absent.
Anything passed on the command line still wins over CONFIG (handy for
one-off overrides without editing the file). Keys set to None fall back
to the upstream default in src/rvln/mininav/interface.py.
"""

import sys
from pathlib import Path

# ============================================================================
# CONFIG -- edit these for your run, then `python scripts/run_hardware.py`.
# ============================================================================

CONFIG = {
    # ---- Task ------------------------------------------------------------
    # Mission instruction (None -> prompted on stdin at startup).
    "instruction": None,
    # Starting world pose: "x,y,z,yaw_deg".
    "initial_position": "0,0,0,0",
    # Run-output directory (None -> results/hardware/).
    "results_dir": None,

    # ---- Camera (set exactly one source) ---------------------------------
    # Local cv2 device index (USB / V4L2). Default 4 = MiniNav USB cam.
    "camera": 4,
    # HTTP frame source, e.g. "http://127.0.0.1:8081/frame" for the mock.
    "camera_url": None,
    # GStreamer pipeline (Jetson MIPI-CSI). Must end in 'appsink'.
    "camera_pipeline": None,
    # Capture target rate (None -> default DEFAULT_CAMERA_FPS).
    "fps": None,
    "camera_retries": None,
    "camera_init_timeout": None,

    # ---- Recording -------------------------------------------------------
    # Persist frames + per-step metadata under run_dir/. Default ON for
    # hardware runs since post-flight inspection wants both the per-step
    # PNGs and recording_log.jsonl alongside playback.mp4.
    "record": True,
    # When recording, throttle log entries to this rate.
    "record_fps": None,

    # ---- Control server (boieng wire) ------------------------------------
    "control_host": None,        # default: 192.168.0.101
    "control_port": None,        # default: 8080
    "control_retries": None,
    "control_retry_sleep": None,
    "command_dt_s": None,

    # ---- OpenVLA ---------------------------------------------------------
    "openvla_predict_url": None,  # default: http://127.0.0.1:5007/predict

    # ---- LTL planner / subgoal converter ---------------------------------
    "llm_model": None,           # default: DEFAULT_LLM_MODEL

    # ---- Goal-adherence monitor (TIME MODE ONLY) -------------------------
    "monitor_model": None,                     # default: DEFAULT_VLM_MODEL
    "diary_check_interval_s": None,            # seconds between checkpoints
    "max_steps_per_subgoal": None,
    "max_seconds_per_subgoal": None,
    "max_corrections": None,
    "stall_window": None,
    "stall_threshold": None,
    "stall_completion_floor": None,

    # ---- Pose source (exactly one of these) -----------------------------
    # dead_reckoning defaults ON since most hardware runs don't have a live
    # odometry feed wired up. To switch to real odometry, set odom_http_url
    # or odom_udp_port AND flip dead_reckoning to False -- they are
    # mutually exclusive and the run aborts if both are configured.
    "odom_http_url": None,
    "odom_udp_host": None,
    "odom_udp_port": None,
    "odom_stale_timeout_s": None,
    "odom_poll_hz": None,
    "dead_reckoning": True,

    # ---- Misc -----------------------------------------------------------
    "extra_env_file": None,
    "log_level": "INFO",
    # Shortcut: True == --log_level DEBUG. Same effect as passing -v on the CLI.
    "verbose": False,
}

# argparse uses dashes for these flags but Python identifiers can't have
# dashes, so we keep underscore keys in CONFIG and translate at flag-build
# time.
_DASHED_FLAGS = {
    "dead_reckoning": "dead-reckoning",
    "extra_env_file": "extra-env-file",
}


def _build_argv(cfg: dict) -> list[str]:
    """Translate CONFIG into argv flags for rvln.mininav.interface.main.

    Hardware always runs time-based monitoring, so --diary-mode is forced
    to 'time' here; remove it from CONFIG to avoid surprising overrides.
    """
    argv: list[str] = ["--diary-mode", "time"]
    for key, value in cfg.items():
        if value is None:
            continue
        flag = "--" + _DASHED_FLAGS.get(key, key)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv


# src/ layout: allow `python scripts/run_hardware.py` without `pip install -e .`
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.mininav.interface import main


if __name__ == "__main__":
    # CONFIG first so CLI overrides win (argparse takes the last value for
    # repeated options). --help still works because it's a CLI arg.
    sys.argv = [sys.argv[0]] + _build_argv(CONFIG) + sys.argv[1:]
    main()
