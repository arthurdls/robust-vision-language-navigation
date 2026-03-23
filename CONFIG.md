# rvln-adls configuration

## Layout

- **config/envs/** – Optional JSON configs for each Unreal env (binary path, map, agents). Not used by the current scripts; only `config/uav_flow_envs/` is.
- **config/uav_flow_envs/** – Overlay configs for UAV-Flow-Eval (e.g. `Track/DowntownWest.json`) so the real environment uses the Linux binary under `envs/UnrealZoo-UE4/` without editing UAV-Flow. `UnrealEnv` must point at `envs/` where `UnrealZoo-UE4` lives.
- **config/defaults.yaml** – Optional default env id (e.g. `default_env_id: UnrealTrack-DowntownWest-ContinuousColor-v0`). Not required by the current scripts.
- **envs/** – Unreal env binaries. Set `UnrealEnv` to this directory (or leave unset; the sim sets it to `repo_root/envs` by default). Put the UnrealZoo-UE4 download here (e.g. `envs/UnrealZoo-UE4/Collection_v4_LinuxNoEditor/...`).
- **tasks/uav_flow_tasks/** – Task JSONs for the UAV-Flow evaluator. Populate by copying from `UAV-Flow/UAV-Flow-Eval/test_jsons/` if needed. `start_openvla_sim.py` reads from here.
- **tasks/ltl_tasks/** – Task JSONs for the LTL runner. `run_openvla_ltl.py` reads from here (e.g. `--task first_task.json` or `--run_all_tasks`).
- **tasks/goal_adherence_tasks/** – Task JSONs for the goal adherence runner. Each file defines a single subgoal with `subgoal`, `initial_pos`, `max_steps`, `diary_check_interval`, and optional `notes`. See schema below.
- **results/** – Output root. UAV-Flow sim writes to `results/uav_flow_results/`. LTL runs write to `results/ltl_results/run_<timestamp>/`. Goal adherence experiments write to `results/goal_adherence_results/<task_name>/`. Playback auto-detects under `results/` or you pass a directory.
- **weights/** – Model checkpoints. Put the OpenVLA checkpoint (HF format) in `weights/` or a subdir (e.g. `weights/OpenVLA-UAV/`). If you use a subdir, the server script auto-detects it when given `weights/`. If the repo uses Git LFS for `*.safetensors`, run `git lfs pull` in the checkout so the shard files are present.
- **scripts/start_openvla_server.py** – Start the OpenVLA server from repo root; uses `weights/` by default (if it contains one subdir with `config.json`, that subdir is used). Override with `--model-dir`, `--port` (default 5007), or `--gpu-id` (default 0). Loads the original UAV-Flow server code (UAV-Flow/OpenVLA-UAV/vla-scripts/openvla_act.py) with no modifications.
- **scripts/sim_common.py** – Shared simulation utilities (constants, env setup, camera helpers, action execution) extracted from `run_openvla_ltl.py` and `run_openvla_repl.py`.

## ai_framework modules

New modules live alongside the existing (unchanged) LTL code:

```
ai_framework/src/modules/
  ltl_planner.py              # LTL symbolic planner (unchanged)
  llm_user_interface.py       # NL-to-LTL conversion (unchanged)
  goal_monitor.py             # Single-image goal monitor for LTL runner (unchanged)
  vision_model.py             # Feasibility checker & planner (unchanged)
  diary_monitor.py            # NEW: LiveDiaryMonitor (real-time diary-based subgoal monitor)
  subgoal_converter.py        # NEW: SubgoalConverter (NL subgoal -> OpenVLA instruction)
  utils/
    base_llm_providers.py     # LLMFactory, OpenAIProvider, GeminiProvider (unchanged)
    other_utils.py            # Misc utilities (unchanged)
    vision_utils.py           # NEW: frame grid, sampling, VLM image queries (provider-agnostic)
```

### LiveDiaryMonitor (`diary_monitor.py`)

Real-time diary-based subgoal completion monitor with supervisor capabilities. Every N steps it builds a local 2-frame grid (what changed?) and a global sampled grid (is the subgoal complete?), maintaining a running diary of changes. Returns one of:

- **stop** – subgoal is complete (early stopping to prevent overshoot)
- **continue** – drone is making progress
- **override** – drone is off-track; replace the OpenVLA instruction
- **command** – supervisor mode corrective instruction (issued on convergence)

General completion criteria are built into the system prompt (movement, approach, visual search, altitude, traversal). No per-task stopping criteria needed.

### SubgoalConverter (`subgoal_converter.py`)

Converts natural language subgoals (e.g., "Turn right until you see the red car") into short imperative OpenVLA instructions (e.g., "turn right"). Strips stopping conditions while preserving spatial references. Called once per task at run start. Uses LLMFactory.

### vision_utils (`utils/vision_utils.py`)

Provider-agnostic vision utilities: `sample_frames_every_n()`, `get_ordered_frames_from_dir()`, `build_frame_grid()`, `query_vlm()`. All VLM calls go through `LLMFactory`/`BaseLLM` so the same code works with any provider.

## Running the simulator

1. Start the OpenVLA server (from repo root, conda env `rvln-server`):  
   `python scripts/start_openvla_server.py`  
   (Uses `weights/` by default; optional: `--port 5007 --gpu-id 0`.)

2. Run the UAV-Flow sim (from repo root, conda env `rvln-sim`):  
   `python scripts/start_openvla_sim.py`  
   (Reads from `tasks/uav_flow_tasks/`, writes to `results/uav_flow_results/`. Pass `-p`, `-m`, etc. to override port or max steps; other args are forwarded to batch_run_act_all.)

3. View results:  
   `python scripts/playback_fpv.py`  
   (Auto-detects latest LTL run under `results/ltl_results/` or UAV-Flow plots under `results/`; or pass a directory, e.g. `results/ltl_results/run_2026_02_23_15_53_54/`.)

## LTL runner (run_openvla_ltl.py)

Same server as above; then run the LTL-based control loop (subgoals + goal monitoring, frames saved under `results/ltl_results/`):

- **Single task from tasks/ltl_tasks/:**  
  `python scripts/run_openvla_ltl.py --task first_task.json`
- **All tasks in tasks/ltl_tasks/:**  
  `python scripts/run_openvla_ltl.py --run_all_tasks`
- **Ad-hoc command (requires --initial-position):**  
  `python scripts/run_openvla_ltl.py -c "Go to the red building..." --initial-position 100,100,100,61`

Uses `ai_framework` for `LLM_User_Interface` and the LTL planner. To update the LTL/LLM stack: replace or update the `ai_framework` directory (and the `spot` dependency).

**Utility:** `scripts/probe_cameras.py` – list cameras and save a frame from each; outputs to `probe_results/<camera_number>/`.

## Goal adherence runner (run_goal_adherence.py)

Tests individual subgoal completion (not full long-horizon plans). For each task, automatically runs N trials without LLM assistance and N trials with LLM diary monitoring.

**Baseline (no LLM):** sends the raw subgoal string directly to OpenVLA as the instruction. Stops on convergence (small-change detection) or max_steps. No conversion, no monitoring.

**LLM-assisted:** SubgoalConverter extracts the OpenVLA instruction. LiveDiaryMonitor provides:
- Early stopping when the subgoal is achieved (overshoot prevention)
- Instruction override when off-track
- Supervisor mode on convergence: issues corrective commands to OpenVLA (e.g., "turn right", "turn left slightly") up to `--max-corrections` times

### Usage

- **Single task:**  
  `python scripts/run_goal_adherence.py --task turn_right_until_red_car.json`
- **All tasks:**  
  `python scripts/run_goal_adherence.py --run-all`
- **Custom model (default: GPT-4o):**  
  `python scripts/run_goal_adherence.py --task turn_right_until_red_car.json --model gpt-5`
- **Custom runs per condition (default: 3):**  
  `python scripts/run_goal_adherence.py --run-all --runs 5`
- **Custom max corrections (default: 10):**  
  `python scripts/run_goal_adherence.py --run-all --max-corrections 15`

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

Fields:
- `task_name` – identifier for results directories
- `subgoal` – the natural language subgoal (as the LTL planner would produce). In baseline runs, sent directly to OpenVLA as-is. In LLM runs, fed to SubgoalConverter and LiveDiaryMonitor.
- `initial_pos` – `[x, y, z, yaw]` (placeholder `[0,0,0,0]` until scouted with `scout_locations.py`)
- `max_steps` – hard step limit per run
- `diary_check_interval` – how many steps between diary checkpoints (N)
- `notes` – optional setup/context notes (not sent to any model)

### Results layout

```
results/goal_adherence_results/
  <task_name>/
    no_llm_run_01/
      frames/
      trajectory_log.json
      run_info.json
    llm_run_01/
      frames/
      trajectory_log.json
      run_info.json
      diary_artifacts/
        checkpoint_0010/ (grid_local.png, grid_global.png, prompts, responses)
        checkpoint_0020/
        convergence_000/ (prompt, response when supervisor intervenes)
        ...
      diary_summary.json
```

`run_info.json` captures: task config, mode (no_llm/llm), instruction sent to OpenVLA, model used (if LLM), total steps, stop reason, instruction overrides, corrections used, timestamps.

## Offline frame analysis (scripts/analyze_frames.py)

CLI tool for offline diary-based frame analysis on saved run directories:

- **Build and view a frame grid (no API call):**  
  `python scripts/analyze_frames.py --dir results/ltl_results/run_2026_03_23/ --grid-only -`
- **Temporal analysis (query the VLM about what's happening):**  
  `python scripts/analyze_frames.py --dir results/ltl_results/run_2026_03_23/ --n 10`
- **Diary-based subtask completion check:**  
  `python scripts/analyze_frames.py --dir results/ltl_results/run_2026_03_23/ --subtask "Move past the traffic light" --n 10`

Default model: GPT-4o (`--model gpt-4o`). Use `--model gpt-5` for the latest model.
