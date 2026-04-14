"""
Centralized path constants and environment variable loading for the rvln project.

All path resolution is relative to the repository root, auto-detected from
this file's location at src/rvln/paths.py.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Repository root: two levels up from src/rvln/paths.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Installed package root (src/rvln)
_RVLN_PKG = Path(__file__).resolve().parent

# Environment / secrets
ENV_FILE = REPO_ROOT / ".env"
ENV_VARS_FILE = REPO_ROOT / "ai_framework" / ".env_vars"  # legacy location

# Unreal scene JSON overlays for gym_unrealcv (Linux env_bin paths, etc.)
DOWNTOWN_OVERLAY_JSON = _RVLN_PKG / "sim" / "scenes" / "Track" / "DowntownWest.json"
DOWNTOWN_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"

# Downloaded Unreal binaries root (gitignored; tools/download_simulator.py)
UNREAL_ENV_ROOT = REPO_ROOT / "runtime" / "unreal"
# Legacy alias: gym / older docs refer to a top-level "envs" tree
ENVS_DIR = UNREAL_ENV_ROOT

# Batch eval script (run via runpy from scripts/run_eval.py)
EVAL_DIR = _RVLN_PKG / "eval"
BATCH_SCRIPT = EVAL_DIR / "batch_runner.py"
# Working directory when invoking the batch runner (relative paths like debug.jpg)
BATCH_RUN_CWD = REPO_ROOT
UAV_FLOW_EVAL = BATCH_RUN_CWD  # legacy name from pre-integration layout

# Runtime directories (gitignored, populated by tools/)
WEIGHTS_DIR = REPO_ROOT / "weights"
RESULTS_DIR = REPO_ROOT / "results"

# Task directories
TASKS_DIR = REPO_ROOT / "tasks"
SYSTEM_TASKS_DIR = TASKS_DIR / "system"
LTL_TASKS_DIR = TASKS_DIR / "ltl"
GOAL_ADHERENCE_TASKS_DIR = TASKS_DIR / "goal_adherence"
UAV_FLOW_TASKS_DIR = TASKS_DIR / "uav_flow"

# Sim / server defaults
DEFAULT_SERVER_PORT = 5007
DEFAULT_TIME_DILATION = 10
DEFAULT_SEED = 0
DEFAULT_INITIAL_POSITION = "-600,-1270,128,61"
DRONE_CAM_ID = 5
PROPRIO_LEN = 4


def load_env_vars() -> None:
    """Load API keys from .env or legacy .env_vars into os.environ.

    Supports shell-style files with ``export KEY=value`` or plain ``KEY=value`` lines.
    Existing environment variables are NOT overwritten.
    """
    for env_path in (ENV_FILE, ENV_VARS_FILE):
        if not env_path.exists():
            continue
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    if key:
                        os.environ.setdefault(key, value)
        except Exception as e:
            logger.warning("Could not load %s: %s", env_path, e)
