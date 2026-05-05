"""
Centralized default configuration for the rvln project.

All tunable defaults live here so they can be changed in one place.
"""

import os

# ---------------------------------------------------------------------------
# LLM / VLM model defaults
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = "gpt-5.4"
DEFAULT_VLM_MODEL = "gpt-5.4"
DEFAULT_LLM_FALLBACK_MODEL = "gpt-4o-mini"
DEFAULT_VLM_FALLBACK_MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# Diary monitoring
# ---------------------------------------------------------------------------
DEFAULT_DIARY_MODE = "frame"  # "frame" (sync, step-based) or "time" (async, time-based)
DEFAULT_HARDWARE_DIARY_MODE = "time"  # hardware default: time-based (async) cadence

# Frame-mode budget / checkpoint interval
DEFAULT_MAX_STEPS_PER_SUBGOAL = 500
DEFAULT_DIARY_CHECK_INTERVAL = 10

# Time-mode budget / checkpoint interval
DEFAULT_MAX_SECONDS_PER_SUBGOAL = 120.0
DEFAULT_DIARY_CHECK_INTERVAL_S = 3.0

# Shared
DEFAULT_MAX_CORRECTIONS = 20
DEFAULT_STALL_WINDOW = 5
DEFAULT_STALL_THRESHOLD = 0.05
DEFAULT_STALL_COMPLETION_FLOOR = 0.5

# ---------------------------------------------------------------------------
# Movement convergence thresholds
# ---------------------------------------------------------------------------
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_STEPS: int = 10

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
DEFAULT_SERVER_PORT = 5007
DEFAULT_SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
DEFAULT_SIM_HOST = os.environ.get("SIM_HOST", "127.0.0.1")
DEFAULT_SIM_PORT = int(os.environ.get("SIM_PORT", "9000"))
DEFAULT_SIM_API_PORT = int(os.environ.get("SIM_API_PORT", "9001"))
DEFAULT_SIM_CONTROLLER_PORT = int(os.environ.get("SIM_CONTROLLER_PORT", "9002"))
DEFAULT_TIME_DILATION = 10
DEFAULT_SEED = 0
PROPRIO_LEN = 4

# ---------------------------------------------------------------------------
# OpenVLA server
# ---------------------------------------------------------------------------
DEFAULT_GPU_ID = 0
DEFAULT_DEVICE = "cuda"
DEFAULT_UNNORM_KEY = "sim"
DEFAULT_DO_SAMPLE = False

# ---------------------------------------------------------------------------
# LTL runner
# ---------------------------------------------------------------------------
DEFAULT_MAX_STEPS = 100

# ---------------------------------------------------------------------------
# Goal adherence
# ---------------------------------------------------------------------------
DEFAULT_RUNS_PER_CONDITION = 3
DEFAULT_GA_MAX_CORRECTIONS = 10

# ---------------------------------------------------------------------------
# Hardware interface (MiniNav)
# ---------------------------------------------------------------------------
DEFAULT_CONTROL_HOST = "192.168.0.101"
DEFAULT_CONTROL_PORT = 8080
DEFAULT_CONTROL_RETRIES = 10
DEFAULT_CONTROL_RETRY_SLEEP = 2.0
DEFAULT_CAMERA_RETRIES = 15
DEFAULT_CAMERA_INIT_TIMEOUT = 8.0
DEFAULT_CAMERA_FPS = 30
DEFAULT_COMMAND_DT_S = 0.1
DEFAULT_ODOM_STALE_TIMEOUT_S = 1.0
DEFAULT_ODOM_UDP_HOST = "0.0.0.0"
DEFAULT_ODOM_UDP_PORT = 0
DEFAULT_OPENVLA_PREDICT_URL = "http://127.0.0.1:5007/predict"

# ---------------------------------------------------------------------------
# Mock frame server
# ---------------------------------------------------------------------------
DEFAULT_FRAME_PORT = 8081
DEFAULT_FRAME_SIZE = 640
DEFAULT_FRAME_GLOB = "**/frames/*.png"
DEFAULT_FRAME_SAMPLE_CAP = 200

# ---------------------------------------------------------------------------
# Batch runner / evaluation
# ---------------------------------------------------------------------------
IMG_INPUT_SIZE = (224, 224)
SLEEP_SHORT_S: float = 1.0
SLEEP_AFTER_RESET_S: float = 2.0
