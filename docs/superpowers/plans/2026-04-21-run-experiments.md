# `scripts/run_experiments.py` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified experiment runner (`scripts/run_experiments.py`) that orchestrates all ablation studies, multi-trial replications, and sensitivity sweeps needed for CoRL 2026 / RA-L / ICRA 2027 submission, producing structured results that map directly to the paper's tables.

**Architecture:** The script defines experiments as declarative configuration dicts (experiment name, condition name, number of runs, which components are enabled/disabled, model choices, parameter overrides). A top-level orchestrator iterates over experiment configs, delegates each run to the existing `run_integrated_control_loop` (for multi-subgoal integration experiments) or `_run_single_ga` (for single-subgoal goal-adherence experiments), and writes per-run results into a structured directory tree. A separate aggregation module reads the results tree and computes per-experiment summary tables (mean/std success rate, steps, corrections, VLM calls, completion percentages) that can be directly pasted into LaTeX. New task JSON files for mission diversity are authored as part of this plan.

**Tech Stack:** Python 3.10+, existing `rvln` package (ai, sim, eval modules), argparse CLI, JSON task/result files.

---

## Experiment Design Rationale

The paper (long_draft.md SS9.2) identifies the experiments needed for acceptance at a top venue. Each run involves an Unreal Engine simulation + OpenVLA inference + VLM API calls, so we must be economical. Here is the minimum experiment set that addresses every major reviewer concern:

### Experiment 1: Component Ablation on the Primary Mission (most critical)

**Purpose:** Isolate the contribution of each component (LTL planner, SubgoalConverter, DiaryMonitor).

Four conditions, 5 runs each = 20 runs:

| Condition | LTL Planner | SubgoalConverter | DiaryMonitor | Description |
|-----------|-------------|------------------|--------------|-------------|
| `full_system` | yes | yes | yes | The complete pipeline |
| `no_diary` | yes | yes | no | LTL decomposition but native convergence only |
| `no_ltl` | no | no | yes | Raw instruction to single DiaryMonitor |
| `raw_baseline` | no | no | no | Raw instruction to OpenVLA, native convergence |

Task: `tasks/system/first_task.json` (the 6-subgoal mission from SS7.2).

**Why 5 runs:** Enough for mean +/- std. 3 is too few for confidence intervals; 10 doubles cost with marginal statistical gain. The paper can report "mean (std) over N=5 trials."

### Experiment 2: Single-Subgoal Goal Adherence Ablation (second most critical)

**Purpose:** Isolate the DiaryMonitor's contribution at the subgoal level, independent of LTL planning.

Two conditions (baseline vs. diary-monitored), 5 runs each, 6 existing tasks = 60 runs.

Tasks: All 6 files in `tasks/goal_adherence/`.

**Why this matters:** Cleanly separates the diary monitor's value from the LTL planner's value. Reviewers will want both experiments (1 for the integrated system, 2 for the monitor in isolation).

### Experiment 3: VLM Model Sensitivity Sweep

**Purpose:** Answer "what if I can't afford gpt-4o?" and characterize the cost-quality frontier.

3 monitor models x 3 runs each = 9 runs on the primary mission (full system only):
- `gpt-4o` (production)
- `gpt-4o-mini` (cheap)
- `gemini-2.0-flash` (alternative provider)

### Experiment 4: Mission Diversity

**Purpose:** Show generalization beyond the single primary mission.

Full system on 4 new missions x 3 runs each = 12 runs. New missions vary in:
- Number of subgoals (3, 4, 5, 5)
- Subgoal type mix (pass-by, reach, turn-until, land, above, between)
- Starting positions (different areas of Downtown West)

### Experiment 5: Diary Parameter Sensitivity

**Purpose:** Characterize the cost-quality frontier for `diary_check_interval`.

3 intervals x 3 runs each = 9 runs on the primary mission (full system):
- `diary_check_interval=5` (more frequent, more VLM calls)
- `diary_check_interval=10` (production default)
- `diary_check_interval=20` (less frequent, fewer VLM calls)

### Total Run Budget

| Experiment | Runs |
|-----------|------|
| 1. Component ablation | 20 |
| 2. Goal adherence ablation | 60 |
| 3. VLM model sensitivity | 9 |
| 4. Mission diversity | 12 |
| 5. Diary parameter sensitivity | 9 |
| **Total** | **110** |

---

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `scripts/run_experiments.py` | CLI entry point: parses args, builds experiment configs, orchestrates runs, delegates to existing control loops |
| `scripts/experiment_configs.py` | Declarative experiment definitions (each experiment is a list of condition dicts); also defines the 4 new mission task dicts for Experiment 4 |
| `scripts/aggregate_results.py` | Post-hoc aggregation: reads the results tree, computes summary tables (mean/std per condition), outputs to stdout and LaTeX-ready TSV |
| `tasks/system/second_task.json` | New 3-subgoal mission for diversity |
| `tasks/system/third_task.json` | New 4-subgoal mission for diversity |
| `tasks/system/fourth_task.json` | New 5-subgoal mission for diversity |
| `tasks/system/fifth_task.json` | New 5-subgoal mission for diversity |

### Existing files to read (not modify)

| File | Why |
|------|-----|
| `scripts/run_integration.py` | Reuse `run_integrated_control_loop`, `_run_subgoal`, `_load_task`, task loading logic |
| `scripts/run_goal_adherence.py` | Reuse `_run_single_ga`, `_run_task_experiments`, `_load_ga_task` |
| `src/rvln/paths.py` | Path constants, defaults |
| `src/rvln/sim/env_setup.py` | Sim setup functions |
| `src/rvln/ai/diary_monitor.py` | DiaryMonitor (used as-is) |
| `src/rvln/ai/ltl_planner.py` | LTL planner (used as-is) |
| `src/rvln/ai/subgoal_converter.py` | SubgoalConverter (used as-is) |

### Results directory structure (output)

```
results/experiments/
  experiment_1_component_ablation/
    full_system/
      run_01/  run_02/  ...  run_05/     (each has run_info.json, trajectory_log.json, frames/, subgoal_*/...)
    no_diary/
      run_01/  ...  run_05/
    no_ltl/
      run_01/  ...  run_05/
    raw_baseline/
      run_01/  ...  run_05/
    summary.json                          (aggregated: mean/std per condition per metric)
  experiment_2_goal_adherence/
    approach_person_ahead/
      baseline/  run_01/ ... run_05/
      diary_monitored/  run_01/ ... run_05/
    approach_white_building/
      ...
    summary.json
  experiment_3_vlm_sensitivity/
    gpt-4o/  run_01/ ... run_03/
    gpt-4o-mini/  run_01/ ... run_03/
    gemini-2.0-flash/  run_01/ ... run_03/
    summary.json
  experiment_4_mission_diversity/
    second_task/  run_01/ ... run_03/
    third_task/   run_01/ ... run_03/
    fourth_task/  run_01/ ... run_03/
    fifth_task/   run_01/ ... run_03/
    summary.json
  experiment_5_diary_params/
    interval_05/  run_01/ ... run_03/
    interval_10/  run_01/ ... run_03/
    interval_20/  run_01/ ... run_03/
    summary.json
```

---

## Task 1: Author New Mission Task JSONs for Experiment 4

**Files:**
- Create: `tasks/system/second_task.json`
- Create: `tasks/system/third_task.json`
- Create: `tasks/system/fourth_task.json`
- Create: `tasks/system/fifth_task.json`

These missions must use valid initial positions in the Downtown West scene and reference landmarks that exist in the environment. The existing tasks use these known-good areas:

- `first_task.json`: `[-7500, 848, 396, -177]` (streetlamps, cars, traffic lights)
- LTL tasks: `[-4795, 301, 349, 175]`, `[12358, 2974, 339, -94]`, `[-181, 7331, 876, -89]`
- Goal adherence tasks: various positions across the map

We design 4 new missions that vary subgoal count and type mix:

- [ ] **Step 1: Create `tasks/system/second_task.json` (3 subgoals, simple)**

```json
{
  "instruction": "Turn left until you see the white building, then go to the white building, then land.",
  "initial_pos": [-86, 8532, 559, -65],
  "max_steps_per_subgoal": 300,
  "diary_check_interval": 10,
  "max_corrections": 15
}
```

This starts at the same position as the `approach_white_building` goal-adherence task (known to have the white building visible after a turn). 3 subgoals: turn-until-detect, reach, land.

- [ ] **Step 2: Create `tasks/system/third_task.json` (4 subgoals, mixed types)**

```json
{
  "instruction": "Go above the pergola, then descend to go between the tree and the streetlight, then go past the streetlight, then turn right until you see a person.",
  "initial_pos": [2217, -1183, 415, 12],
  "max_steps_per_subgoal": 300,
  "diary_check_interval": 10,
  "max_corrections": 15
}
```

Starts at the pergola area (from goal-adherence tasks). 4 subgoals: above, between, pass-by, turn-until-detect.

- [ ] **Step 3: Create `tasks/system/fourth_task.json` (5 subgoals, navigation-heavy)**

```json
{
  "instruction": "Go past the first tree, then turn right until you see the black car, then go to the black car, then turn left until you see a streetlight, then go past the streetlight.",
  "initial_pos": [-4795, 301, 349, 175],
  "max_steps_per_subgoal": 300,
  "diary_check_interval": 10,
  "max_corrections": 15
}
```

Starts at the LTL first_task position. 5 subgoals: pass-by, turn-until-detect, reach, turn-until-detect, pass-by.

- [ ] **Step 4: Create `tasks/system/fifth_task.json` (5 subgoals, altitude + spatial)**

```json
{
  "instruction": "Descend 2 meters, then go forward until you see the umbrella, then go between the tree and the umbrella, then ascend 3 meters, then turn right until you see the street lamp and approach it.",
  "initial_pos": [-181, 7331, 876, -89],
  "max_steps_per_subgoal": 300,
  "diary_check_interval": 10,
  "max_corrections": 15
}
```

Starts at the LTL third_task position (high altitude, trees and umbrellas nearby). 5 subgoals: descend, forward-until-detect, between, ascend, turn-until-detect + reach.

- [ ] **Step 5: Commit**

```bash
git add tasks/system/second_task.json tasks/system/third_task.json tasks/system/fourth_task.json tasks/system/fifth_task.json
git commit -m "Add 4 new system tasks for mission diversity experiments"
```

---

## Task 2: Create `scripts/experiment_configs.py` (Declarative Experiment Definitions)

**Files:**
- Create: `scripts/experiment_configs.py`

This file defines each experiment as a structured config that the runner can iterate over. It has no side effects and imports nothing from the sim stack, so it can be loaded without the Unreal environment.

- [ ] **Step 1: Write `scripts/experiment_configs.py`**

```python
#!/usr/bin/env python3
"""
Declarative experiment configurations for run_experiments.py.

Each experiment is a dict with:
  - name: str (experiment directory name)
  - description: str
  - type: "integration" | "goal_adherence"
  - conditions: list of condition dicts

Each condition dict contains:
  - name: str (condition directory name)
  - runs: int (number of trials)
  - task_file: str | None (relative to tasks/system/ or tasks/goal_adherence/)
  - task_files: list[str] | None (for goal_adherence with multiple tasks)
  
  For integration experiments:
  - use_ltl: bool (whether to use LTL planner for decomposition)
  - use_diary: bool (whether to use LiveDiaryMonitor)
  - use_converter: bool (whether to use SubgoalConverter)
  - llm_model: str (model for LTL planner + SubgoalConverter)
  - monitor_model: str (model for DiaryMonitor VLM queries)
  - diary_check_interval: int
  - max_steps_per_subgoal: int
  - max_corrections: int

  For goal_adherence experiments:
  - use_llm: bool
  - model: str
  - max_corrections: int
"""

from typing import Any, Dict, List

# Shared defaults matching the paper's production config
_PROD_LLM = "gpt-4o-mini"
_PROD_MONITOR = "gpt-4o"
_PROD_CHECK_INTERVAL = 10
_PROD_MAX_STEPS = 300
_PROD_MAX_CORRECTIONS = 15
_PRIMARY_TASK = "first_task.json"


def _integration_condition(
    name: str,
    runs: int,
    task_file: str = _PRIMARY_TASK,
    use_ltl: bool = True,
    use_diary: bool = True,
    use_converter: bool = True,
    llm_model: str = _PROD_LLM,
    monitor_model: str = _PROD_MONITOR,
    diary_check_interval: int = _PROD_CHECK_INTERVAL,
    max_steps_per_subgoal: int = _PROD_MAX_STEPS,
    max_corrections: int = _PROD_MAX_CORRECTIONS,
) -> Dict[str, Any]:
    return {
        "name": name,
        "runs": runs,
        "task_file": task_file,
        "use_ltl": use_ltl,
        "use_diary": use_diary,
        "use_converter": use_converter,
        "llm_model": llm_model,
        "monitor_model": monitor_model,
        "diary_check_interval": diary_check_interval,
        "max_steps_per_subgoal": max_steps_per_subgoal,
        "max_corrections": max_corrections,
    }


def _ga_condition(
    name: str,
    runs: int,
    use_llm: bool,
    model: str = _PROD_MONITOR,
    max_corrections: int = _PROD_MAX_CORRECTIONS,
) -> Dict[str, Any]:
    return {
        "name": name,
        "runs": runs,
        "use_llm": use_llm,
        "model": model,
        "max_corrections": max_corrections,
    }


# ── Experiment 1: Component Ablation ──────────────────────────────────────

EXPERIMENT_1_COMPONENT_ABLATION: Dict[str, Any] = {
    "name": "experiment_1_component_ablation",
    "description": (
        "Ablation study on the primary 6-subgoal mission. "
        "Four conditions isolate the LTL planner and DiaryMonitor contributions. "
        "Maps to paper SS7.5 baselines 1-3 and the full system."
    ),
    "type": "integration",
    "conditions": [
        _integration_condition(
            name="full_system",
            runs=5,
        ),
        _integration_condition(
            name="no_diary",
            runs=5,
            use_diary=False,
        ),
        _integration_condition(
            name="no_ltl",
            runs=5,
            use_ltl=False,
            use_converter=False,
        ),
        _integration_condition(
            name="raw_baseline",
            runs=5,
            use_ltl=False,
            use_diary=False,
            use_converter=False,
        ),
    ],
}


# ── Experiment 2: Single-Subgoal Goal Adherence ──────────────────────────

_GA_TASK_FILES = [
    "approach_person_ahead.json",
    "approach_white_building.json",
    "go_above_pergola.json",
    "go_between_tree_and_streetlight.json",
    "move_through_pergola.json",
    "turn_right_until_red_car.json",
]

EXPERIMENT_2_GOAL_ADHERENCE: Dict[str, Any] = {
    "name": "experiment_2_goal_adherence",
    "description": (
        "Single-subgoal diary monitor ablation. "
        "Baseline (raw subgoal to OpenVLA) vs diary-monitored, "
        "5 runs each on all 6 goal-adherence tasks. "
        "Isolates the DiaryMonitor's contribution independent of LTL planning."
    ),
    "type": "goal_adherence",
    "task_files": _GA_TASK_FILES,
    "conditions": [
        _ga_condition(name="baseline", runs=5, use_llm=False),
        _ga_condition(name="diary_monitored", runs=5, use_llm=True),
    ],
}


# ── Experiment 3: VLM Model Sensitivity ──────────────────────────────────

EXPERIMENT_3_VLM_SENSITIVITY: Dict[str, Any] = {
    "name": "experiment_3_vlm_sensitivity",
    "description": (
        "Monitor model sweep on the primary mission (full system). "
        "Tests gpt-4o (production), gpt-4o-mini (cheap), and gemini-2.0-flash (alt provider)."
    ),
    "type": "integration",
    "conditions": [
        _integration_condition(
            name="gpt-4o",
            runs=3,
            monitor_model="gpt-4o",
        ),
        _integration_condition(
            name="gpt-4o-mini",
            runs=3,
            monitor_model="gpt-4o-mini",
        ),
        _integration_condition(
            name="gemini-2.0-flash",
            runs=3,
            monitor_model="gemini-2.0-flash",
        ),
    ],
}


# ── Experiment 4: Mission Diversity ──────────────────────────────────────

_DIVERSITY_TASKS = [
    "second_task.json",
    "third_task.json",
    "fourth_task.json",
    "fifth_task.json",
]

EXPERIMENT_4_MISSION_DIVERSITY: Dict[str, Any] = {
    "name": "experiment_4_mission_diversity",
    "description": (
        "Full system on 4 additional missions varying subgoal count (3-5) and type mix. "
        "3 runs each. Demonstrates generalization beyond the primary mission."
    ),
    "type": "integration",
    "conditions": [
        _integration_condition(
            name=task_file.replace(".json", ""),
            runs=3,
            task_file=task_file,
        )
        for task_file in _DIVERSITY_TASKS
    ],
}


# ── Experiment 5: Diary Parameter Sensitivity ────────────────────────────

EXPERIMENT_5_DIARY_PARAMS: Dict[str, Any] = {
    "name": "experiment_5_diary_params",
    "description": (
        "Diary check interval sweep on the primary mission (full system). "
        "Tests interval=5 (frequent), 10 (production), 20 (sparse)."
    ),
    "type": "integration",
    "conditions": [
        _integration_condition(
            name="interval_05",
            runs=3,
            diary_check_interval=5,
        ),
        _integration_condition(
            name="interval_10",
            runs=3,
            diary_check_interval=10,
        ),
        _integration_condition(
            name="interval_20",
            runs=3,
            diary_check_interval=20,
        ),
    ],
}


# ── Registry ─────────────────────────────────────────────────────────────

ALL_EXPERIMENTS: List[Dict[str, Any]] = [
    EXPERIMENT_1_COMPONENT_ABLATION,
    EXPERIMENT_2_GOAL_ADHERENCE,
    EXPERIMENT_3_VLM_SENSITIVITY,
    EXPERIMENT_4_MISSION_DIVERSITY,
    EXPERIMENT_5_DIARY_PARAMS,
]

EXPERIMENT_BY_NAME: Dict[str, Dict[str, Any]] = {
    exp["name"]: exp for exp in ALL_EXPERIMENTS
}
```

- [ ] **Step 2: Commit**

```bash
git add scripts/experiment_configs.py
git commit -m "Add declarative experiment configurations for ablation studies"
```

---

## Task 3: Create `scripts/run_experiments.py` (Main Runner)

**Files:**
- Create: `scripts/run_experiments.py`

This is the main entry point. It:
1. Parses CLI args to select which experiment(s) to run
2. Sets up the sim environment once (shared across all runs)
3. Iterates over experiment conditions and runs
4. Delegates to the existing control loops with the right flags
5. Skips runs whose `run_info.json` already exists (resumable)
6. Logs progress and saves per-run results in the structured directory tree

The key design challenge is handling the ablation conditions. For the `no_diary` and `raw_baseline` conditions, we need to run the integration loop or a simplified version that skips the diary monitor. Rather than duplicating the control loop, we modify the call parameters:

- **`full_system`**: Calls `run_integrated_control_loop` as-is.
- **`no_diary`**: Calls a modified version that replaces `_run_subgoal` with a diary-free version that only uses native convergence.
- **`no_ltl`**: Sends the raw instruction as a single subgoal to a single `_run_subgoal` call (with diary).
- **`raw_baseline`**: Sends the raw instruction to OpenVLA with no decomposition and no diary (uses `_run_single_ga` from goal adherence in baseline mode).

- [ ] **Step 1: Write `scripts/run_experiments.py`**

```python
#!/usr/bin/env python3
"""
Unified experiment runner for ablation studies.

Orchestrates all experiments needed for the paper (component ablation,
goal adherence, VLM sensitivity, mission diversity, parameter sweeps).

Each experiment is defined declaratively in experiment_configs.py.
Results are saved in a structured directory tree under results/experiments/.
Runs are resumable: existing run_info.json files are skipped unless
--override-runs is set.

OpenVLA server must be running: python scripts/start_server.py

Usage (from repo root):
  # Run all experiments
  python scripts/run_experiments.py --run-all

  # Run a specific experiment
  python scripts/run_experiments.py --experiment experiment_1_component_ablation

  # Run a specific condition within an experiment
  python scripts/run_experiments.py --experiment experiment_1_component_ablation --condition full_system

  # List available experiments
  python scripts/run_experiments.py --list

  # Override existing results
  python scripts/run_experiments.py --experiment experiment_1_component_ablation --override-runs

  # Aggregate results after runs complete
  python scripts/aggregate_results.py --experiment experiment_1_component_ablation
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from rvln.paths import (
    BATCH_SCRIPT,
    DEFAULT_SERVER_PORT,
    DEFAULT_SEED,
    DEFAULT_TIME_DILATION,
    DOWNTOWN_ENV_ID,
    DRONE_CAM_ID,
    REPO_ROOT,
    UAV_FLOW_EVAL,
)
from rvln.sim.env_setup import (
    apply_action_poses,
    import_batch_module,
    load_env_vars,
    normalize_initial_pos,
    parse_position,
    relative_pose_to_world,
    set_drone_cam_and_get_image,
    setup_env_and_imports,
    setup_sim_env,
    state_for_openvla,
)

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from experiment_configs import ALL_EXPERIMENTS, EXPERIMENT_BY_NAME

EXPERIMENTS_RESULTS_DIR = REPO_ROOT / "results" / "experiments"
SYSTEM_TASKS_DIR = REPO_ROOT / "tasks" / "system"
GA_TASKS_DIR = REPO_ROOT / "tasks" / "goal_adherence"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task loading (reused from run_integration.py and run_goal_adherence.py)
# ---------------------------------------------------------------------------

def _load_system_task(path: Path) -> Dict[str, Any]:
    """Load a system task JSON."""
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
    return {
        "instruction": instruction,
        "initial_pos": [float(x) for x in initial_pos],
        "max_steps_per_subgoal": int(data.get("max_steps_per_subgoal", 300)),
        "diary_check_interval": int(data.get("diary_check_interval", 10)),
        "max_corrections": int(data.get("max_corrections", 15)),
    }


def _load_ga_task(path: Path) -> Dict[str, Any]:
    """Load a goal adherence task JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    required = ["task_name", "subgoal", "initial_pos", "max_steps", "diary_check_interval"]
    for key in required:
        if key not in data:
            raise ValueError(f"Task JSON missing required field: {key}")
    return data


def _sanitize_name(text: str, max_len: int = 40) -> str:
    clean = text.lower().replace(" ", "_")
    safe = "".join(c for c in clean if c.isalnum() or c == "_")
    return safe[:max_len] or "subgoal"


# ---------------------------------------------------------------------------
# Integration run variants
# ---------------------------------------------------------------------------

def _run_integration_full_system(
    env: Any,
    batch: Any,
    task: Dict[str, Any],
    server_url: str,
    run_dir: Path,
    condition: Dict[str, Any],
    drone_cam_id: int,
) -> Dict[str, Any]:
    """Run the full integrated pipeline (LTL + converter + diary monitor).

    Delegates to run_integrated_control_loop from run_integration.py.
    """
    from run_integration import run_integrated_control_loop

    return run_integrated_control_loop(
        env=env,
        batch=batch,
        task={
            **task,
            "diary_check_interval": condition["diary_check_interval"],
            "max_steps_per_subgoal": condition["max_steps_per_subgoal"],
            "max_corrections": condition["max_corrections"],
        },
        server_url=server_url,
        run_dir=run_dir,
        llm_model=condition["llm_model"],
        monitor_model=condition["monitor_model"],
        drone_cam_id=drone_cam_id,
    )


def _run_integration_no_diary(
    env: Any,
    batch: Any,
    task: Dict[str, Any],
    server_url: str,
    run_dir: Path,
    condition: Dict[str, Any],
    drone_cam_id: int,
) -> Dict[str, Any]:
    """Run LTL decomposition + SubgoalConverter but no diary monitor.

    Uses native convergence detection only. Each subgoal runs until
    the drone converges (small-change detection) or max_steps is reached,
    then the LTL planner advances.
    """
    import math
    from rvln.ai.llm_interface import LLM_User_Interface
    from rvln.ai.ltl_planner import LTL_Symbolic_Planner
    from rvln.ai.subgoal_converter import SubgoalConverter

    instruction = task["instruction"]
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps = condition["max_steps_per_subgoal"]

    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3],
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[4] - 180,
    )
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    start_ts = datetime.now().isoformat()

    llm_interface = LLM_User_Interface(model=condition["llm_model"])
    planner = LTL_Symbolic_Planner(llm_interface)
    planner.plan_from_natural_language(instruction)

    ltl_plan = {
        "ltl_nl_formula": llm_interface.ltl_nl_formula.get("ltl_nl_formula", ""),
        "pi_predicates": dict(planner.pi_map),
    }

    converter = SubgoalConverter(model=condition["llm_model"])

    subgoal_summaries: List[Dict[str, Any]] = []
    trajectory_log: List[Dict[str, Any]] = []
    total_frame_count = 0
    subgoal_index = 0

    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[4]

    current_subgoal = planner.get_next_predicate()
    if current_subgoal is None:
        raise RuntimeError("LTL planning produced no subgoals.")

    SMALL_DELTA_POS = 3.0
    SMALL_DELTA_YAW = 1.0

    while current_subgoal is not None:
        subgoal_index += 1
        safe_name = _sanitize_name(current_subgoal)
        subgoal_dir = run_dir / f"subgoal_{subgoal_index:02d}_{safe_name}"
        subgoal_dir.mkdir(parents=True, exist_ok=True)

        converted_instruction = converter.convert(current_subgoal)

        logger.info(
            "--- [no_diary] Subgoal %d: '%s' -> '%s' ---",
            subgoal_index, current_subgoal, converted_instruction,
        )

        batch.reset_model(server_url)

        current_pose = [0.0, 0.0, 0.0, 0.0]
        last_pose = None
        small_count = 0
        stop_reason = "max_steps"
        subgoal_steps = 0

        for step in range(max_steps):
            batch.set_cam(env)
            image = set_drone_cam_and_get_image(env, drone_cam_id)
            if image is None:
                stop_reason = "no_image"
                break

            global_idx = total_frame_count + step
            frame_path = frames_dir / f"frame_{global_idx:06d}.png"
            try:
                import cv2
                cv2.imwrite(str(frame_path), image)
            except Exception:
                pass

            response = batch.send_prediction_request(
                image=Image.fromarray(image),
                proprio=state_for_openvla(current_pose),
                instr=converted_instruction.strip().lower(),
                server_url=server_url,
            )

            if response is None:
                stop_reason = "no_response"
                subgoal_steps = step
                break

            action_poses = response.get("action")
            if not isinstance(action_poses, list) or len(action_poses) == 0:
                stop_reason = "empty_action"
                subgoal_steps = step
                break

            try:
                new_image, current_pose, steps_added = apply_action_poses(
                    env, action_poses,
                    origin_x, origin_y, origin_z, origin_yaw,
                    batch.set_cam,
                    trajectory_log=trajectory_log,
                    sleep_s=0.1,
                    drone_cam_id=drone_cam_id,
                )
            except Exception as e:
                logger.error("Action error at step %d: %s", step, e)
                stop_reason = "action_error"
                subgoal_steps = step
                break

            subgoal_steps = step + 1

            if last_pose is not None:
                diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
                if all(d < SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < SMALL_DELTA_YAW:
                    small_count += 1
                else:
                    small_count = 0
                if small_count >= batch.ACTION_SMALL_STEPS:
                    stop_reason = "convergence"
                    break
            last_pose = list(current_pose)
        else:
            stop_reason = "max_steps"
            subgoal_steps = max_steps

        total_frame_count += subgoal_steps

        next_origin_x, next_origin_y, next_origin_z, next_origin_yaw = (
            relative_pose_to_world(origin_x, origin_y, origin_z, origin_yaw, current_pose)
        )

        subgoal_result = {
            "subgoal": current_subgoal,
            "converted_instruction": converted_instruction,
            "total_steps": subgoal_steps,
            "stop_reason": stop_reason,
            "corrections_used": 0,
            "last_completion_pct": None,
            "peak_completion": None,
            "vlm_calls": 0,
            "next_origin": [next_origin_x, next_origin_y, next_origin_z, next_origin_yaw],
        }
        subgoal_summaries.append(subgoal_result)

        with open(subgoal_dir / "subgoal_result.json", "w") as f:
            json.dump(subgoal_result, f, indent=2)

        logger.info(
            "Subgoal %d finished: stop_reason=%s, steps=%d",
            subgoal_index, stop_reason, subgoal_steps,
        )

        planner.advance_state(current_subgoal)
        origin_x, origin_y, origin_z, origin_yaw = (
            next_origin_x, next_origin_y, next_origin_z, next_origin_yaw,
        )
        current_subgoal = planner.get_next_predicate()

    end_ts = datetime.now().isoformat()

    with open(run_dir / "trajectory_log.json", "w") as f:
        json.dump(trajectory_log, f, indent=2)

    total_steps_all = sum(s["total_steps"] for s in subgoal_summaries)
    run_info = {
        "task": task,
        "condition": "no_diary",
        "llm_model": condition["llm_model"],
        "monitor_model": None,
        "ltl_plan": ltl_plan,
        "subgoal_count": subgoal_index,
        "subgoal_summaries": subgoal_summaries,
        "total_steps": total_steps_all,
        "total_vlm_calls": 0,
        "total_corrections": 0,
        "start_time": start_ts,
        "end_time": end_ts,
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    return run_info


def _run_integration_no_ltl(
    env: Any,
    batch: Any,
    task: Dict[str, Any],
    server_url: str,
    run_dir: Path,
    condition: Dict[str, Any],
    drone_cam_id: int,
) -> Dict[str, Any]:
    """Run raw instruction with a single DiaryMonitor (no LTL decomposition).

    The entire mission instruction is treated as a single subgoal.
    The DiaryMonitor supervises; convergence corrections are applied.
    No SubgoalConverter is used (the raw instruction goes to OpenVLA).
    """
    from run_integration import _run_subgoal

    initial_pos = normalize_initial_pos(task["initial_pos"])
    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3],
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[4] - 180,
    )
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)

    start_ts = datetime.now().isoformat()

    trajectory_log: List[Dict[str, Any]] = []

    subgoal_dir = run_dir / "subgoal_01_full_mission"

    subgoal_result = _run_subgoal(
        env=env,
        batch=batch,
        server_url=server_url,
        subgoal_nl=task["instruction"],
        monitor_model=condition["monitor_model"],
        check_interval=condition["diary_check_interval"],
        max_steps=condition["max_steps_per_subgoal"],
        max_corrections=condition["max_corrections"],
        origin_x=initial_pos[0],
        origin_y=initial_pos[1],
        origin_z=initial_pos[2],
        origin_yaw=initial_pos[4],
        drone_cam_id=drone_cam_id,
        frames_dir=frames_dir,
        subgoal_dir=subgoal_dir,
        frame_offset=0,
        trajectory_log=trajectory_log,
    )

    end_ts = datetime.now().isoformat()

    with open(run_dir / "trajectory_log.json", "w") as f:
        json.dump(trajectory_log, f, indent=2)

    run_info = {
        "task": task,
        "condition": "no_ltl",
        "llm_model": None,
        "monitor_model": condition["monitor_model"],
        "ltl_plan": None,
        "subgoal_count": 1,
        "subgoal_summaries": [subgoal_result],
        "total_steps": subgoal_result["total_steps"],
        "total_vlm_calls": subgoal_result.get("vlm_calls", 0),
        "total_corrections": subgoal_result.get("corrections_used", 0),
        "start_time": start_ts,
        "end_time": end_ts,
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    return run_info


def _run_integration_raw_baseline(
    env: Any,
    batch: Any,
    task: Dict[str, Any],
    server_url: str,
    run_dir: Path,
    condition: Dict[str, Any],
    drone_cam_id: int,
) -> Dict[str, Any]:
    """Run raw instruction directly on OpenVLA with native convergence only.

    No LTL, no SubgoalConverter, no DiaryMonitor. This is the weakest
    baseline: OpenVLA receives the full multi-step instruction as a
    single string and runs until native convergence or max_steps.
    """
    initial_pos = normalize_initial_pos(task["initial_pos"])
    max_steps = condition["max_steps_per_subgoal"]

    run_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env.unwrapped.unrealcv.set_obj_location(
        env.unwrapped.player_list[0], initial_pos[0:3],
    )
    env.unwrapped.unrealcv.set_rotation(
        env.unwrapped.player_list[0], initial_pos[4] - 180,
    )
    batch.set_cam(env)
    time.sleep(batch.SLEEP_AFTER_RESET_S)
    batch.reset_model(server_url)

    start_ts = datetime.now().isoformat()
    instruction = task["instruction"].strip().lower()

    current_pose = [0.0, 0.0, 0.0, 0.0]
    last_pose = None
    small_count = 0
    trajectory_log: List[Dict[str, Any]] = []
    stop_reason = "max_steps"
    total_steps = 0

    SMALL_DELTA_POS = 3.0
    SMALL_DELTA_YAW = 1.0

    origin_x, origin_y, origin_z = initial_pos[0], initial_pos[1], initial_pos[2]
    origin_yaw = initial_pos[4]

    for step in range(max_steps):
        batch.set_cam(env)
        image = set_drone_cam_and_get_image(env, drone_cam_id)
        if image is None:
            stop_reason = "no_image"
            break

        frame_path = frames_dir / f"frame_{step:06d}.png"
        try:
            import cv2
            cv2.imwrite(str(frame_path), image)
        except Exception:
            pass

        response = batch.send_prediction_request(
            image=Image.fromarray(image),
            proprio=state_for_openvla(current_pose),
            instr=instruction,
            server_url=server_url,
        )

        if response is None:
            stop_reason = "no_response"
            total_steps = step
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            stop_reason = "empty_action"
            total_steps = step
            break

        try:
            new_image, current_pose, steps_added = apply_action_poses(
                env, action_poses,
                origin_x, origin_y, origin_z, origin_yaw,
                batch.set_cam,
                trajectory_log=trajectory_log,
                sleep_s=0.1,
                drone_cam_id=drone_cam_id,
            )
        except Exception as e:
            logger.error("Action error at step %d: %s", step, e)
            stop_reason = "action_error"
            total_steps = step
            break

        total_steps = step + 1

        if last_pose is not None:
            diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
            if all(d < SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < SMALL_DELTA_YAW:
                small_count += 1
            else:
                small_count = 0
            if small_count >= batch.ACTION_SMALL_STEPS:
                stop_reason = "convergence"
                break
        last_pose = list(current_pose)
    else:
        stop_reason = "max_steps"
        total_steps = max_steps

    end_ts = datetime.now().isoformat()

    with open(run_dir / "trajectory_log.json", "w") as f:
        json.dump(trajectory_log, f, indent=2)

    run_info = {
        "task": task,
        "condition": "raw_baseline",
        "llm_model": None,
        "monitor_model": None,
        "ltl_plan": None,
        "subgoal_count": 0,
        "subgoal_summaries": [],
        "total_steps": total_steps,
        "total_vlm_calls": 0,
        "total_corrections": 0,
        "stop_reason": stop_reason,
        "start_time": start_ts,
        "end_time": end_ts,
    }
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    return run_info


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def _run_integration_condition(
    env: Any,
    batch: Any,
    condition: Dict[str, Any],
    server_url: str,
    condition_dir: Path,
    drone_cam_id: int,
    override_runs: bool,
) -> None:
    """Run all trials for one integration experiment condition."""
    task_file = condition["task_file"]
    task_path = SYSTEM_TASKS_DIR / task_file
    if not task_path.exists():
        logger.error("Task file not found: %s", task_path)
        return
    task = _load_system_task(task_path)

    use_ltl = condition["use_ltl"]
    use_diary = condition["use_diary"]

    for run_idx in range(1, condition["runs"] + 1):
        run_name = f"run_{run_idx:02d}"
        run_dir = condition_dir / run_name
        run_info_path = run_dir / "run_info.json"

        if not override_runs and run_info_path.exists():
            logger.info("  Skipping %s/%s (already exists)", condition["name"], run_name)
            continue

        logger.info("  Running %s/%s ...", condition["name"], run_name)

        try:
            if use_ltl and use_diary:
                run_info = _run_integration_full_system(
                    env, batch, task, server_url, run_dir, condition, drone_cam_id,
                )
            elif use_ltl and not use_diary:
                run_info = _run_integration_no_diary(
                    env, batch, task, server_url, run_dir, condition, drone_cam_id,
                )
            elif not use_ltl and use_diary:
                run_info = _run_integration_no_ltl(
                    env, batch, task, server_url, run_dir, condition, drone_cam_id,
                )
            else:
                run_info = _run_integration_raw_baseline(
                    env, batch, task, server_url, run_dir, condition, drone_cam_id,
                )

            logger.info(
                "    Done: %d steps, %d subgoals",
                run_info.get("total_steps", 0),
                run_info.get("subgoal_count", 0),
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error("    FAILED: %s", e, exc_info=True)


def _run_ga_condition(
    env: Any,
    batch: Any,
    condition: Dict[str, Any],
    task_files: List[str],
    server_url: str,
    experiment_dir: Path,
    drone_cam_id: int,
    override_runs: bool,
) -> None:
    """Run all trials for one goal-adherence condition across all tasks."""
    from run_goal_adherence import _run_single_ga

    for task_file in task_files:
        task_path = GA_TASKS_DIR / task_file
        if not task_path.exists():
            logger.error("Task file not found: %s", task_path)
            continue
        task = _load_ga_task(task_path)
        task_name = task["task_name"]

        for run_idx in range(1, condition["runs"] + 1):
            run_name = f"run_{run_idx:02d}"
            run_dir = experiment_dir / task_name / condition["name"] / run_name
            run_info_path = run_dir / "run_info.json"

            if not override_runs and run_info_path.exists():
                logger.info(
                    "  Skipping %s/%s/%s (already exists)",
                    task_name, condition["name"], run_name,
                )
                continue

            logger.info(
                "  Running %s/%s/%s ...", task_name, condition["name"], run_name,
            )

            try:
                run_info = _run_single_ga(
                    env=env,
                    task=task,
                    batch=batch,
                    server_url=server_url,
                    run_dir=run_dir,
                    use_llm=condition["use_llm"],
                    model=condition["model"],
                    drone_cam_id=drone_cam_id,
                    max_corrections=condition["max_corrections"],
                )
                logger.info(
                    "    Done: %d steps, stop=%s",
                    run_info.get("total_steps", 0),
                    run_info.get("stop_reason", "?"),
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("    FAILED: %s", e, exc_info=True)


def run_experiment(
    env: Any,
    batch: Any,
    experiment: Dict[str, Any],
    server_url: str,
    results_base: Path,
    drone_cam_id: int,
    override_runs: bool,
    condition_filter: Optional[str] = None,
) -> None:
    """Run all conditions (or a filtered subset) of one experiment."""
    exp_name = experiment["name"]
    exp_type = experiment["type"]
    exp_dir = results_base / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", exp_name)
    logger.info("  %s", experiment["description"])
    logger.info("=" * 60)

    conditions = experiment["conditions"]
    if condition_filter:
        conditions = [c for c in conditions if c["name"] == condition_filter]
        if not conditions:
            logger.error(
                "Condition '%s' not found in experiment '%s'. Available: %s",
                condition_filter, exp_name,
                [c["name"] for c in experiment["conditions"]],
            )
            return

    for condition in conditions:
        cond_name = condition["name"]
        logger.info("-" * 40)
        logger.info("CONDITION: %s (%d runs)", cond_name, condition["runs"])
        logger.info("-" * 40)

        if exp_type == "integration":
            condition_dir = exp_dir / cond_name
            condition_dir.mkdir(parents=True, exist_ok=True)
            _run_integration_condition(
                env, batch, condition, server_url,
                condition_dir, drone_cam_id, override_runs,
            )
        elif exp_type == "goal_adherence":
            task_files = experiment.get("task_files", [])
            _run_ga_condition(
                env, batch, condition, task_files,
                server_url, exp_dir, drone_cam_id, override_runs,
            )
        else:
            logger.error("Unknown experiment type: %s", exp_type)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_env_vars()
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for ablation studies and sweeps",
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--run-all",
        action="store_true",
        help="Run all experiments",
    )
    action.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run a specific experiment by name",
    )
    action.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )

    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Run only this condition within the selected experiment",
    )

    parser.add_argument("-e", "--env_id", default=DOWNTOWN_ENV_ID)
    parser.add_argument("-t", "--time_dilation", type=int, default=DEFAULT_TIME_DILATION)
    parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("-p", "--server_port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument(
        "-o", "--results_dir",
        default=str(EXPERIMENTS_RESULTS_DIR),
        help="Base directory for experiment results",
    )
    parser.add_argument("--use-default-cam", action="store_true")
    parser.add_argument("--override-runs", action="store_true",
                        help="Re-run experiments even if results exist")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:\n")
        for exp in ALL_EXPERIMENTS:
            print(f"  {exp['name']}")
            print(f"    {exp['description']}")
            print(f"    Type: {exp['type']}")
            print(f"    Conditions: {[c['name'] for c in exp['conditions']]}")
            total_runs = sum(c["runs"] for c in exp["conditions"])
            if exp["type"] == "goal_adherence":
                total_runs *= len(exp.get("task_files", []))
            print(f"    Total runs: {total_runs}")
            print()
        return

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    if args.condition and not args.experiment:
        parser.error("--condition requires --experiment")

    experiments_to_run: List[Dict[str, Any]] = []
    if args.run_all:
        experiments_to_run = ALL_EXPERIMENTS
    elif args.experiment:
        if args.experiment not in EXPERIMENT_BY_NAME:
            logger.error(
                "Unknown experiment: '%s'. Use --list to see available experiments.",
                args.experiment,
            )
            sys.exit(1)
        experiments_to_run = [EXPERIMENT_BY_NAME[args.experiment]]

    if not BATCH_SCRIPT.exists():
        logger.error("batch_run_act_all.py not found at %s", BATCH_SCRIPT)
        sys.exit(1)

    setup_env_and_imports()
    batch = import_batch_module()
    os.chdir(str(UAV_FLOW_EVAL))

    server_url = f"http://127.0.0.1:{args.server_port}/predict"
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    env = setup_sim_env(args.env_id, int(args.time_dilation), int(args.seed), batch)

    drone_cam_id = DRONE_CAM_ID
    if not args.use_default_cam:
        from rvln.sim.env_setup import interactive_camera_select
        logger.info("Camera selection: pick the camera for OpenVLA.")
        drone_cam_id = interactive_camera_select(
            env, [0, 0, 300, 0], batch,
        )

    for experiment in experiments_to_run:
        try:
            run_experiment(
                env=env,
                batch=batch,
                experiment=experiment,
                server_url=server_url,
                results_base=results_base,
                drone_cam_id=drone_cam_id,
                override_runs=args.override_runs,
                condition_filter=args.condition,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted. Partial results saved.")
            break

    env.close()
    logger.info("All experiments complete. Results in %s", results_base)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_experiments.py
git commit -m "Add unified experiment runner for ablation studies"
```

---

## Task 4: Create `scripts/aggregate_results.py` (Results Aggregation)

**Files:**
- Create: `scripts/aggregate_results.py`

This script reads the results directory tree after experiments complete and produces summary tables. It can be run independently of the simulator.

- [ ] **Step 1: Write `scripts/aggregate_results.py`**

```python
#!/usr/bin/env python3
"""
Aggregate experiment results into summary tables.

Reads the structured results tree produced by run_experiments.py and
computes per-condition summary statistics (mean, std, min, max) for
key metrics. Outputs to stdout and optionally to a LaTeX-ready TSV.

Usage:
  # Summarize all experiments
  python scripts/aggregate_results.py

  # Summarize a specific experiment
  python scripts/aggregate_results.py --experiment experiment_1_component_ablation

  # Export to TSV for LaTeX
  python scripts/aggregate_results.py --experiment experiment_1_component_ablation --tsv results/tables/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "results" / "experiments"


def _collect_run_infos(condition_dir: Path) -> List[Dict[str, Any]]:
    """Collect all run_info.json dicts from run_01/, run_02/, etc."""
    infos = []
    for run_dir in sorted(condition_dir.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        info_path = run_dir / "run_info.json"
        if info_path.exists():
            with open(info_path) as f:
                infos.append(json.load(f))
    return infos


def _subgoal_success_rate(run_info: Dict[str, Any]) -> Optional[float]:
    """Compute fraction of subgoals completed (stop_reason=monitor_complete)."""
    summaries = run_info.get("subgoal_summaries", [])
    if not summaries:
        return None
    completed = sum(
        1 for s in summaries
        if s.get("stop_reason") == "monitor_complete"
    )
    return completed / len(summaries)


def _mean_std(values: List[float]) -> str:
    if not values:
        return "N/A"
    arr = np.array(values)
    return f"{arr.mean():.2f} +/- {arr.std():.2f}"


def _summarize_integration_condition(
    condition_name: str,
    run_infos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize one integration experiment condition."""
    n = len(run_infos)
    if n == 0:
        return {"condition": condition_name, "runs": 0}

    total_steps = [r.get("total_steps", 0) for r in run_infos]
    vlm_calls = [r.get("total_vlm_calls", 0) for r in run_infos]
    corrections = [r.get("total_corrections", 0) for r in run_infos]
    subgoal_counts = [r.get("subgoal_count", 0) for r in run_infos]

    success_rates = [
        sr for sr in [_subgoal_success_rate(r) for r in run_infos]
        if sr is not None
    ]

    peak_completions = []
    for r in run_infos:
        for s in r.get("subgoal_summaries", []):
            hwm = s.get("peak_completion")
            if hwm is not None:
                peak_completions.append(hwm)

    incomplete_hwms = []
    for r in run_infos:
        for s in r.get("subgoal_summaries", []):
            if s.get("stop_reason") != "monitor_complete" and s.get("peak_completion") is not None:
                incomplete_hwms.append(s["peak_completion"])

    return {
        "condition": condition_name,
        "runs": n,
        "subgoal_success_rate": _mean_std(success_rates),
        "total_steps": _mean_std([float(x) for x in total_steps]),
        "vlm_calls": _mean_std([float(x) for x in vlm_calls]),
        "corrections": _mean_std([float(x) for x in corrections]),
        "subgoal_count": _mean_std([float(x) for x in subgoal_counts]),
        "mean_incomplete_hwm": _mean_std(incomplete_hwms) if incomplete_hwms else "N/A",
    }


def _summarize_ga_condition(
    condition_name: str,
    task_run_infos: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Summarize one goal-adherence condition across all tasks."""
    all_steps = []
    all_vlm_calls = []
    all_corrections = []
    completed_count = 0
    total_count = 0

    per_task = {}
    for task_name, infos in sorted(task_run_infos.items()):
        task_completed = sum(
            1 for r in infos
            if r.get("stop_reason") in ("llm_stopped", "monitor_complete")
        )
        task_total = len(infos)
        completed_count += task_completed
        total_count += task_total

        all_steps.extend(r.get("total_steps", 0) for r in infos)
        all_vlm_calls.extend(r.get("vlm_calls", 0) for r in infos)
        all_corrections.extend(r.get("corrections_used", 0) for r in infos)

        per_task[task_name] = {
            "runs": task_total,
            "completed": task_completed,
            "success_rate": f"{task_completed / task_total:.2f}" if task_total > 0 else "N/A",
            "mean_steps": _mean_std([float(r.get("total_steps", 0)) for r in infos]),
        }

    overall_sr = completed_count / total_count if total_count > 0 else 0.0

    return {
        "condition": condition_name,
        "total_runs": total_count,
        "overall_success_rate": f"{overall_sr:.2f}",
        "total_steps": _mean_std([float(x) for x in all_steps]),
        "vlm_calls": _mean_std([float(x) for x in all_vlm_calls]),
        "corrections": _mean_std([float(x) for x in all_corrections]),
        "per_task": per_task,
    }


def summarize_experiment(exp_dir: Path) -> None:
    """Print summary for one experiment directory."""
    exp_name = exp_dir.name
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'=' * 60}\n")

    is_ga = "goal_adherence" in exp_name

    if is_ga:
        condition_names = set()
        task_names = set()
        for task_dir in sorted(exp_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name == "summary.json":
                continue
            task_names.add(task_dir.name)
            for cond_dir in sorted(task_dir.iterdir()):
                if cond_dir.is_dir():
                    condition_names.add(cond_dir.name)

        summaries = []
        for cond_name in sorted(condition_names):
            task_run_infos = {}
            for task_name in sorted(task_names):
                cond_dir = exp_dir / task_name / cond_name
                if cond_dir.exists():
                    task_run_infos[task_name] = _collect_run_infos(cond_dir)
            summary = _summarize_ga_condition(cond_name, task_run_infos)
            summaries.append(summary)
            print(f"Condition: {cond_name}")
            print(f"  Total runs: {summary['total_runs']}")
            print(f"  Overall success rate: {summary['overall_success_rate']}")
            print(f"  Steps: {summary['total_steps']}")
            print(f"  VLM calls: {summary['vlm_calls']}")
            print(f"  Corrections: {summary['corrections']}")
            for tn, td in summary.get("per_task", {}).items():
                print(f"  {tn}: {td['completed']}/{td['runs']} success, steps {td['mean_steps']}")
            print()
    else:
        summaries = []
        for cond_dir in sorted(exp_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            infos = _collect_run_infos(cond_dir)
            if not infos:
                continue
            summary = _summarize_integration_condition(cond_dir.name, infos)
            summaries.append(summary)
            print(f"Condition: {summary['condition']}")
            print(f"  Runs: {summary['runs']}")
            print(f"  Subgoal success rate: {summary['subgoal_success_rate']}")
            print(f"  Total steps: {summary['total_steps']}")
            print(f"  VLM calls: {summary['vlm_calls']}")
            print(f"  Corrections: {summary['corrections']}")
            print(f"  Mean incomplete HWM: {summary['mean_incomplete_hwm']}")
            print()

    summary_path = exp_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results into summary tables",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Summarize a specific experiment (default: all)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(EXPERIMENTS_DIR),
        help="Base results directory",
    )
    parser.add_argument(
        "--tsv",
        type=str,
        default=None,
        help="Directory to write LaTeX-ready TSV files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    if args.experiment:
        exp_dir = results_dir / args.experiment
        if not exp_dir.exists():
            print(f"Experiment not found: {exp_dir}")
            sys.exit(1)
        summarize_experiment(exp_dir)
    else:
        for exp_dir in sorted(results_dir.iterdir()):
            if exp_dir.is_dir():
                summarize_experiment(exp_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/aggregate_results.py
git commit -m "Add results aggregation script for experiment summaries"
```

---

## Task 5: Validate the Full Pipeline (Dry Run)

**Files:**
- Read: `scripts/run_experiments.py`
- Read: `scripts/experiment_configs.py`
- Read: `scripts/aggregate_results.py`

Verify the scripts are importable and the CLI works without launching the sim.

- [ ] **Step 1: Test `--list` command**

Run:
```bash
cd /home/arthurdls/SuperUROP/rvln-adls && python scripts/run_experiments.py --list
```

Expected output: A table listing all 5 experiments with their conditions and run counts. No sim startup, no errors.

- [ ] **Step 2: Test experiment_configs imports cleanly**

Run:
```bash
cd /home/arthurdls/SuperUROP/rvln-adls && python -c "from scripts.experiment_configs import ALL_EXPERIMENTS; print(f'{len(ALL_EXPERIMENTS)} experiments, {sum(sum(c[\"runs\"] for c in e[\"conditions\"]) for e in ALL_EXPERIMENTS)} base runs')"
```

Expected: `5 experiments, 55 base runs` (the 55 is the base count before multiplying goal-adherence tasks).

- [ ] **Step 3: Test aggregate_results runs on empty dir**

Run:
```bash
cd /home/arthurdls/SuperUROP/rvln-adls && mkdir -p results/experiments/test_empty && python scripts/aggregate_results.py --experiment test_empty && rm -r results/experiments/test_empty
```

Expected: Prints the experiment header, finds no conditions, writes an empty summary.json, no crash.

- [ ] **Step 4: Verify new task JSONs are valid**

Run:
```bash
cd /home/arthurdls/SuperUROP/rvln-adls && python -c "
import json
from pathlib import Path
for f in sorted(Path('tasks/system').glob('*.json')):
    data = json.loads(f.read_text())
    inst = data.get('instruction', '')[:60]
    pos = data.get('initial_pos', [])
    print(f'{f.name}: pos={pos}, instruction=\"{inst}...\"')
"
```

Expected: Lists all 5 system task files with their positions and instruction previews.

- [ ] **Step 5: Commit (no code changes, just verification)**

No commit needed if everything passes. If any fixes were required, commit them:

```bash
git add -u
git commit -m "Fix issues found during dry-run validation"
```

---

## Summary: Mapping Experiments to Paper Sections

| Experiment | Paper section | Reviewer question it answers |
|-----------|--------------|------------------------------|
| 1. Component ablation | SS7.5 (baselines 1-3 + full system) | "Have you isolated each component's contribution?" |
| 2. Goal adherence ablation | SS7.5 (diary monitor in isolation) | "Does the diary monitor help at the single-subgoal level?" |
| 3. VLM sensitivity | SS9.2 item 4 | "What if I can't afford gpt-4o?" |
| 4. Mission diversity | SS9.2 item 3 | "Does this generalize beyond one mission?" |
| 5. Diary parameter sensitivity | SS9.2 item 9 | "How sensitive is performance to check_interval?" |

### Key metrics per experiment

**Experiment 1 (ablation):**
- Subgoal success rate (primary; expect full_system >> no_diary > no_ltl >> raw_baseline)
- Total steps, corrections, VLM calls (cost)
- Mean high-water-mark on incomplete subgoals (diagnosis quality)

**Experiment 2 (goal adherence):**
- Per-task success rate: diary_monitored vs baseline
- Mean steps to completion (efficiency)
- Per-task breakdown by subgoal type

**Experiment 3 (VLM sensitivity):**
- Success rate vs model (cost-quality frontier)
- VLM calls (same) but corrections quality differs

**Experiment 4 (diversity):**
- Per-mission success rate (generalization)
- Per-subgoal-type success matrix across all missions

**Experiment 5 (diary params):**
- Success rate vs check_interval
- VLM calls vs check_interval (cost-quality)

### Running the experiments

Recommended execution order (most to least critical):

```bash
# 1. Component ablation (most critical, ~20 runs)
python scripts/run_experiments.py --experiment experiment_1_component_ablation --use-default-cam

# 2. Goal adherence ablation (second most critical, ~60 runs)
python scripts/run_experiments.py --experiment experiment_2_goal_adherence --use-default-cam

# 3. VLM sensitivity (cheap, high leverage, ~9 runs)
python scripts/run_experiments.py --experiment experiment_3_vlm_sensitivity --use-default-cam

# 4. Mission diversity (~12 runs)
python scripts/run_experiments.py --experiment experiment_4_mission_diversity --use-default-cam

# 5. Diary params (~9 runs)
python scripts/run_experiments.py --experiment experiment_5_diary_params --use-default-cam

# Aggregate results
python scripts/aggregate_results.py
```
