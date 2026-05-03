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

# Environment / secrets (see load_env_vars for load order)
ENV_FILE = REPO_ROOT / ".env"
ENV_FILE_LOCAL = REPO_ROOT / ".env.local"

# Unreal scene JSON overlays for gym_unrealcv (Linux env_bin paths, etc.)
SCENES_DIR = _RVLN_PKG / "sim" / "scenes"

# Downloaded Unreal binaries root (gitignored; tools/download_simulator.py)
UNREAL_ENV_ROOT = REPO_ROOT / "runtime" / "unreal"
# Legacy alias: gym / older docs refer to a top-level "envs" tree
ENVS_DIR = UNREAL_ENV_ROOT

EVAL_DIR = _RVLN_PKG / "eval"
BATCH_SCRIPT = EVAL_DIR / "batch_runner.py"
# Working directory when invoking the batch runner (relative paths like debug.jpg)
BATCH_RUN_CWD = REPO_ROOT
UAV_FLOW_EVAL = BATCH_RUN_CWD  # alias used by scripts that chdir before running batch_runner

# Runtime directories (gitignored, populated by tools/)
WEIGHTS_DIR = REPO_ROOT / "weights"
RESULTS_DIR = REPO_ROOT / "results"

# Task directories
TASKS_DIR = REPO_ROOT / "tasks"
SYSTEM_TASKS_DIR = TASKS_DIR / "system"
LTL_TASKS_DIR = TASKS_DIR / "ltl"
GOAL_ADHERENCE_TASKS_DIR = TASKS_DIR / "goal_adherence"
UAV_FLOW_TASKS_DIR = TASKS_DIR / "uav_flow"

# Sim / server defaults moved to config.py
from rvln.config import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_SIM_HOST,
    DEFAULT_SIM_PORT,
    DEFAULT_TIME_DILATION,
    PROPRIO_LEN,
)


def _load_env_file(env_path: Path, *, override: bool) -> None:
    """Parse shell-style env file; setdefault unless override (local secrets win)."""
    try:
        with open(env_path, "r", encoding="utf-8") as f:
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
                if not key:
                    continue
                if override:
                    os.environ[key] = value
                else:
                    os.environ.setdefault(key, value)
    except Exception as e:
        logger.warning("Could not load %s: %s", env_path, e)


def load_env_vars(extra_override: Path | str | None = None) -> None:
    """Load API keys from env files into ``os.environ``.

    Order:

    1. ``.env`` — fills missing keys only (shared defaults).
    2. ``.env.local`` — **overrides** keys (recommended for API keys on this machine).
    3. ``extra_override`` — optional path from callers, loaded last with override (CLI / tests).

    Supports ``export KEY=value`` or plain ``KEY=value`` lines.
    """
    seen: set[Path] = set()

    def _one(path: Path, override: bool) -> None:
        if not path.exists():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        _load_env_file(path, override=override)

    _one(ENV_FILE, override=False)
    _one(ENV_FILE_LOCAL, override=True)
    if extra_override is not None:
        _one(Path(extra_override), override=True)
