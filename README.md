# rvln-adls

Neuro-symbolic LTL planner with vision-language supervision for UAV control in simulation.

This system decomposes multi-step natural language instructions into an LTL formula, converts each subgoal into short OpenVLA commands, and supervises execution with a diary-based VLM monitor that can issue corrections. It runs in an Unreal Engine simulation environment via UnrealCV.

## Architecture

```
Instruction: "Go to the red building, then land near the tree"
    |
    v
LTL Planner (LLM -> Spot automaton)
    |
    v  subgoal: "approach the red building"
SubgoalConverter (LLM -> short OpenVLA command)
    |
    v  command: "fly toward building"
OpenVLA Server (VLA model, returns drone actions)
    |
    v  action: [dx, dy, dz, dyaw]
Unreal Sim (UnrealCV gym environment)
    ^
    |  periodic frames
LiveDiaryMonitor (VLM diary: continue / stop / correct)
```

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> rvln-adls && cd rvln-adls
bash tools/setup.sh

# 2. Configure API keys
cp .env.example .env.local
# Edit .env.local with your OpenAI / Google API keys

# 3. Download assets (requires ~20GB disk)
conda activate rvln-sim
pip install -e .
python tools/download_weights.py
python tools/download_simulator.py

# 4. Start OpenVLA server (Terminal 1, needs GPU)
conda activate rvln-server
pip install -e ".[server]"
python scripts/start_server.py

# 5. Run the integrated pipeline (Terminal 2)
conda activate rvln-sim
python scripts/run_integration.py --task first_task.json
```

## Prerequisites

- **CUDA GPU** for the OpenVLA server (tested with A100/4090)
- **conda** (Miniconda or Anaconda)
- **~20GB disk** for model weights + Unreal environment
- **API keys** for OpenAI and/or Google (for LLM/VLM calls), typically in `.env.local`

## Directory Structure

```
rvln-adls/
  src/rvln/               Python package (pip install -e .)
    paths.py              Centralized path constants and env loading
    ai/                   LTL planner, diary monitor, subgoal converter, LLM providers
    sim/                  Unreal sim env setup, pose utilities, scene JSON overlays
    eval/                 Batch evaluation runner, metrics, playback
    server/               OpenVLA inference server
    mininav/              MiniNav real-drone interface
  src/gym_unrealcv/       UnrealCV gym environments (vendored from UAV-Flow)
  scripts/                CLI entry points
  tools/                  Setup and download scripts
  tasks/                  Task JSON files (system, ltl, goal_adherence, uav_flow)
  tests/                  Test suite
  runtime/unreal/         Unreal binaries (gitignored, populated by tools/)
  weights/                Model checkpoints (gitignored, populated by tools/)
  results/                Run outputs (gitignored)
```

The Unreal download target is `runtime/unreal/` (the code sets `UnrealEnv` to that path by default). If you still have data under the old top-level `envs/` folder, move it into `runtime/unreal/` or export `UnrealEnv` to point at your existing tree.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/start_server.py` | Start OpenVLA inference server (Flask, port 5007) |
| `scripts/run_integration.py` | Full LTL + diary monitor pipeline |
| `scripts/run_ltl.py` | LTL-only control loop (no diary) |
| `scripts/run_eval.py` | UAV-Flow batch evaluation |
| `scripts/run_goal_adherence.py` | Single-subgoal goal adherence experiments |
| `scripts/run_repl.py` | Interactive REPL for drone commands |
| `scripts/analyze_frames.py` | Offline VLM frame analysis |
| `scripts/playback.py` | FPV viewer and MP4 encoder |
| `scripts/probe_cameras.py` | Camera diagnostic tool |
| `scripts/scout_locations.py` | Position scouting for task authoring |

## Task JSON Format

Tasks live in `tasks/system/`, `tasks/ltl/`, `tasks/goal_adherence/`, and `tasks/uav_flow/`.

```json
{
  "instruction": "Go to the red building then land near the tree",
  "initial_pos": [-600, -1270, 128, 0, 61],
  "max_steps_per_subgoal": 200,
  "diary_check_interval": 10,
  "max_corrections": 5
}
```

## Conda Environments

Two separate environments are used because the server requires a heavy GPU stack:

- **rvln-sim** (`rvln-sim_env.yml`): Simulation client, LTL planner, LLM/VLM APIs. CPU-only.
- **rvln-server** (`rvln-server_env.yml`): PyTorch + transformers + flash-attn for OpenVLA inference. Requires CUDA GPU.

Both should have the `rvln` package installed:
```bash
conda activate rvln-sim && pip install -e .
conda activate rvln-server && pip install -e ".[server]"
```

Note: The `spot` LTL model checker is only available via conda-forge (not pip). It is included in `rvln-sim_env.yml`.

API keys are read from `.env` then `.env.local` (see `rvln.paths.load_env_vars`); put secrets in `.env.local` so they stay out of shared defaults.

## Vendored Code

`src/gym_unrealcv/` and parts of `src/rvln/eval/` and `src/rvln/server/` are vendored from [UAV-Flow](https://github.com/buaa-colalab/UAV-Flow) (commit `0114801`).
