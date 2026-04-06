# rvln-adls configuration

## Layout

- **config/envs/** – Optional JSON configs for each Unreal env (binary path, map, agents). Not used by the current scripts; only `config/uav_flow_envs/` is.
- **config/uav_flow_envs/** – Overlay configs for UAV-Flow-Eval (e.g. `Track/DowntownWest.json`) so the real environment uses the Linux binary under `envs/UnrealZoo-UE4/` without editing UAV-Flow. `UnrealEnv` must point at `envs/` where `UnrealZoo-UE4` lives.
- **config/defaults.yaml** – Optional default env id (e.g. `default_env_id: UnrealTrack-DowntownWest-ContinuousColor-v0`). Not required by the current scripts.
- **envs/** – Unreal env binaries. Set `UnrealEnv` to this directory (or leave unset; the sim sets it to `repo_root/envs` by default). Put the UnrealZoo-UE4 download here (e.g. `envs/UnrealZoo-UE4/Collection_v4_LinuxNoEditor/...`).
- **tasks/uav_flow_tasks/** – Task JSONs for the UAV-Flow evaluator. Populate by copying from `UAV-Flow/UAV-Flow-Eval/test_jsons/` if needed. `run_uav_flow_eval.py` reads from here.
- **tasks/ltl_tasks/** – Multi-step task JSONs for the LTL-only runner. `run_ltl_planner.py` reads from here (`--task`, `--run_all_tasks`).
- **tasks/system_tasks/** – Task JSONs for the integrated runner (`run_system_integration.py`: LTL planner + diary monitor). Same schema as LTL tasks; optional fields `diary_check_interval`, `max_steps_per_subgoal`, and `max_corrections` apply only here. Keep these separate from `ltl_tasks` when you want different missions or parameters for the full pipeline.
- **tasks/goal_adherence_tasks/** – Task JSONs for the goal adherence runner. Each file defines a single subgoal with `subgoal`, `initial_pos`, `max_steps`, `diary_check_interval`, and optional `notes`. See schema below.
- **results/** – Output root. UAV-Flow sim writes to `results/uav_flow_results/`. LTL runs write to `results/ltl_results/run_<timestamp>/`. Goal adherence experiments write to `results/goal_adherence_results/<task_name>/`. Integrated runs write to `results/integration_results/run_<timestamp>/`. Playback auto-detects under `results/` or you pass a directory.
- **weights/** – Model checkpoints. Put the OpenVLA checkpoint (HF format) in `weights/` or a subdir (e.g. `weights/OpenVLA-UAV/`). If you use a subdir, the server script auto-detects it when given `weights/`. If the repo uses Git LFS for `*.safetensors`, run `git lfs pull` in the checkout so the shard files are present.
- **scripts/** – Orchestration scripts (see sections below).
- **.old_code/** – Archived code from earlier iterations (gitignored).

## ai_framework modules

```
ai_framework/src/modules/
  ltl_planner.py              # LTL symbolic planner (Spot automaton)
  llm_user_interface.py       # NL-to-LTL-NL conversion via LLM
  diary_monitor.py            # LiveDiaryMonitor (real-time diary-based subgoal monitor)
  subgoal_converter.py        # SubgoalConverter (NL subgoal -> OpenVLA instruction)
  vision_model.py             # Feasibility checker & planner
  utils/
    base_llm_providers.py     # LLMFactory, OpenAIProvider, GeminiProvider
    other_utils.py            # LTL formula parsing (parse_ltl_nl, extract_json)
    vision_utils.py           # Frame grid, sampling, VLM image queries (provider-agnostic)
    goal_adherence_utils.py   # Offline diary-based completion checking, prompt templates
```

### LiveDiaryMonitor (`diary_monitor.py`)

Real-time diary-based subgoal completion monitor with supervisor capabilities. Every N steps it builds a local 2-frame grid (what changed?) and a global sampled grid (is the subgoal complete?), maintaining a running diary of changes. Returns one of:

- **stop** – subgoal is complete (early stopping to prevent overshoot)
- **continue** – drone is making progress
- **force_converge** – drone is actively making things worse; force convergence for correction
- **command** – supervisor mode corrective instruction (issued on convergence)

General completion criteria are built into the system prompt (movement, approach, visual search, altitude, traversal). No per-task stopping criteria needed.

### SubgoalConverter (`subgoal_converter.py`)

Converts natural language subgoals (e.g., "Turn right until you see the red car") into short imperative OpenVLA instructions (e.g., "turn right"). Strips stopping conditions while preserving spatial references. Called once per subgoal at execution start. Uses LLMFactory.

### goal_adherence_utils (`utils/goal_adherence_utils.py`)

Prompt templates and offline diary-based workflows: `analyze_temporal_frames()`, `check_subtask_complete_diary()`, `parse_yes_no_response()`, `query_what_changed_between_frames()`. Used by `scripts/analyze_frames.py` for post-hoc analysis of saved runs. Builds on `vision_utils.py`.

### vision_utils (`utils/vision_utils.py`)

Provider-agnostic vision utilities: `sample_frames_every_n()`, `get_ordered_frames_from_dir()`, `build_frame_grid()`, `query_vlm()`. All VLM calls go through `LLMFactory`/`BaseLLM` so the same code works with any provider.

## Scripts

| Script | Purpose |
|--------|---------|
| `start_openvla_server.py` | Start OpenVLA server (Flask, port 5007) |
| `run_uav_flow_eval.py` | Run default UAV-Flow evaluation suite (reads `tasks/uav_flow_tasks/`) |
| `sim_common.py` | Shared sim utilities (env setup, coordinate transforms, action execution) |
| `run_ltl_planner.py` | LTL-only control loop; reads `tasks/ltl_tasks/` |
| `run_goal_adherence.py` | Single-subgoal goal adherence experiments (baseline vs LLM diary) |
| `run_system_integration.py` | **Integrated LTL planner + diary monitor**; reads `tasks/system_tasks/` |
| `run_repl.py` | Interactive REPL for natural-language drone commands |
| `analyze_frames.py` | Offline frame analysis (grid building, VLM temporal queries, diary completion) |
| `playback_fpv.py` | FPV frame viewer and MP4 encoder |
| `probe_cameras.py` | List cameras and save a frame from each |
| `scout_locations.py` | Fly around and record positions for task authoring |

## Running the simulator

1. Start the OpenVLA server (from repo root, conda env `rvln-server`):  
   `python scripts/start_openvla_server.py`  
   (Uses `weights/` by default; optional: `--port 5007 --gpu-id 0`.)

2. Run the default UAV-Flow evaluation suite (from repo root, conda env `rvln-sim`):  
   `python scripts/run_uav_flow_eval.py`  
   (Reads from `tasks/uav_flow_tasks/`, writes to `results/uav_flow_results/`. Pass `-p`, `-m`, etc. to override port or max steps; other args are forwarded to batch_run_act_all.)

3. View results:  
   `python scripts/playback_fpv.py`  
   (Auto-detects latest LTL run under `results/ltl_results/` or UAV-Flow plots under `results/`; or pass a directory.)

## Integrated runner (run_system_integration.py)

Combines the LTL-NL neuro-symbolic planner with the LiveDiaryMonitor for full multi-step execution with per-subgoal diary supervision and corrective commands. This is the primary experiment runner. Task files are under **`tasks/system_tasks/`** (separate from **`tasks/ltl_tasks/`** used by `run_ltl_planner.py`).

For each subgoal from the LTL planner:
1. `SubgoalConverter` rewrites the predicate NL into an OpenVLA instruction.
2. A fresh `LiveDiaryMonitor` supervises execution with periodic VLM checkpoints.
3. On convergence, the monitor issues corrective commands or confirms completion.
4. The planner advances to the next subgoal.

### Usage

- **Single task:**  
  `python scripts/run_system_integration.py --task third_task.json`
- **All tasks:**  
  `python scripts/run_system_integration.py --run_all_tasks`
- **Ad-hoc command:**  
  `python scripts/run_system_integration.py -c "Go to the tree then land" --initial-position -181,7331,876,-89`
- **Custom models:**  
  `python scripts/run_system_integration.py --task first_task.json --llm_model gpt-4o --monitor_model gpt-4o`
- **Custom diary parameters:**  
  `python scripts/run_system_integration.py --task first_task.json --diary-check-interval 10 --max-steps-per-subgoal 200 --max-corrections 10`

### Results layout

```
results/integration_results/
  run_<timestamp>/
    run_info.json              # task config, LTL plan, per-subgoal summaries, timing
    trajectory_log.json        # full trajectory across all subgoals
    frames/                    # all FPV frames (sequentially numbered)
    subgoal_01_<name>/
      diary_summary.json       # diary, overrides, completion %, vlm calls
      diary_artifacts/
        checkpoint_NNNN/       # grid_local.png, grid_global.png, prompts, responses
        convergence_NNN/       # convergence prompt, response, diary snapshot
    subgoal_02_<name>/
      ...
```

## LTL runner (run_ltl_planner.py)

Bare LTL control loop without diary monitoring. Subgoals advance on convergence (pose stall), max_steps, or OpenVLA done signal. For diary-based monitoring, use `run_system_integration.py` instead.

- **Single task from tasks/ltl_tasks/:**  
  `python scripts/run_ltl_planner.py --task first_task.json`
- **All tasks in tasks/ltl_tasks/:**  
  `python scripts/run_ltl_planner.py --run_all_tasks`
- **Ad-hoc command (requires --initial-position):**  
  `python scripts/run_ltl_planner.py -c "Go to the red building..." --initial-position 100,100,100,61`

## Goal adherence runner (run_goal_adherence.py)

Tests individual subgoal completion (not full long-horizon plans). For each task, automatically runs N trials without LLM assistance and N trials with LLM diary monitoring.

**Baseline (no LLM):** sends the raw subgoal string directly to OpenVLA as the instruction. Stops on convergence (small-change detection) or max_steps. No conversion, no monitoring.

**LLM-assisted:** SubgoalConverter extracts the OpenVLA instruction. LiveDiaryMonitor provides:
- Early stopping when the subgoal is achieved (overshoot prevention)
- Supervisor mode on convergence: issues corrective commands to OpenVLA up to `--max-corrections` times

### Usage

- **Single task:**  
  `python scripts/run_goal_adherence.py --task turn_right_until_red_car.json`
- **All tasks:**  
  `python scripts/run_goal_adherence.py --run-all`
- **Custom model (default: GPT-4o):**  
  `python scripts/run_goal_adherence.py --task turn_right_until_red_car.json --model gpt-5`
- **LLM-assisted runs only (skip baseline `no_llm_run_*`):**  
  `python scripts/run_goal_adherence.py --run-all --llm-only`
- **Baseline runs only (skip LLM `llm_run_*`):**  
  `python scripts/run_goal_adherence.py --run-all --baseline-only`

### Task JSON schema (tasks/goal_adherence_tasks/)

```json
{
  "task_name": "turn_right_until_red_car",
  "subgoal": "Turn right until you see the red car",
  "initial_pos": [0, 0, 0, 0],
  "max_steps": 200,
  "diary_check_interval": 10,
  "notes": "Start by looking away from both the red and white car"
}
```

## Offline frame analysis (scripts/analyze_frames.py)

CLI tool for offline diary-based frame analysis on saved run directories:

- **Build and view a frame grid (no API call):**  
  `python scripts/analyze_frames.py --dir results/ltl_results/run_2026_03_23/ --grid-only -`
- **Temporal analysis (query the VLM about what's happening):**  
  `python scripts/analyze_frames.py --dir results/ltl_results/run_2026_03_23/ --n 10`
- **Diary-based subtask completion check:**  
  `python scripts/analyze_frames.py --dir results/ltl_results/run_2026_03_23/ --subtask "Move past the traffic light" --n 10`

Default model: GPT-4o (`--model gpt-4o`). Use `--model gpt-5` for the latest model.
