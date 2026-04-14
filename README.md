# Robust Vision-Language Navigation

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

## Prerequisites

- **CUDA GPU** for the OpenVLA server (tested with A100 / RTX 4090)
- **conda** (Miniconda or Anaconda)
- **~20 GB disk** for model weights + Unreal environment
- **API keys** for OpenAI and/or Google (LLM / VLM calls), placed in `.env.local`

## Quick Start

```bash
# 1. Clone and enter the repo
git clone git@github.com:arthurdls/robust-vision-language-navigation.git
cd robust-vision-language-navigation

# 2. Create both conda envs (rvln-sim + rvln-server) and scaffold .env.local
bash tools/setup.sh

# 3. Install the rvln package in each env
conda activate rvln-sim    && pip install -e .
conda activate rvln-server && pip install -e ".[server]"

# 4. Configure API keys
$EDITOR .env.local          # OPENAI_API_KEY, GOOGLE_API_KEY

# 5. Download assets (~20 GB)
conda activate rvln-sim
python tools/download_weights.py       # -> weights/OpenVLA-UAV/
python tools/download_simulator.py     # -> runtime/unreal/

# 6. Start the OpenVLA server (Terminal 1, GPU required)
conda activate rvln-server
python scripts/start_server.py

# 7. Run the integrated pipeline (Terminal 2)
conda activate rvln-sim
python scripts/run_integration.py --task first_task.json
```

Most commands are also wrapped in the `Makefile` (`make setup`, `make download-weights`, `make server`, `make run`, `make repl`, `make lint`).

## Directory Structure

```
robust-vision-language-navigation/
  src/rvln/               Python package (pip install -e .)
    paths.py              Centralized path constants and env loading
    ai/                   LTL planner, diary monitor, subgoal converter, LLM providers
    sim/                  Unreal sim env setup, pose utilities, scene JSON overlays
    eval/                 Batch evaluation runner, metrics, playback
    server/               OpenVLA inference server
    mininav/              MiniNav real-drone interface
  src/gym_unrealcv/       UnrealCV gym environments (vendored from UAV-Flow)
  scripts/                CLI entry points (see table below)
  tools/                  Setup, weight download, simulator download
  tasks/                  Task JSONs: system/, ltl/, goal_adherence/, uav_flow/, uav_flow_instructions/
  tests/                  Test suite
  runtime/unreal/         Unreal binaries (gitignored, populated by tools/download_simulator.py)
  weights/                Model checkpoints (gitignored, populated by tools/download_weights.py)
  results/                Run outputs (gitignored)
```

The Unreal download target is `runtime/unreal/` (matching `rvln.paths.UNREAL_ENV_ROOT`). If you have existing data under a legacy top-level `envs/` directory, move it into `runtime/unreal/` or export `UnrealEnv` to point at your existing tree.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/start_server.py` | Start the OpenVLA inference server (Flask, port 5007) |
| `scripts/run_integration.py` | Full LTL planner + diary monitor pipeline |
| `scripts/run_ltl.py` | LTL-only control loop (no diary supervision) |
| `scripts/run_goal_adherence.py` | Single-subgoal goal adherence experiments |
| `scripts/run_repl.py` | Interactive REPL for drone commands |
| `scripts/run_old_original_eval.py` | Legacy UAV-Flow batch evaluation runner |
| `scripts/playback.py` | FPV viewer and MP4 encoder for saved runs |
| `scripts/scout_locations.py` | Position scouting helper for task authoring |

## Conda Environments

Two separate environments are used because the OpenVLA server needs a heavy CUDA/PyTorch stack while the simulator client does not:

- **rvln-sim** (`rvln-sim_env.yml`): Simulator client, LTL planner, LLM / VLM API calls. CPU-only, includes the Spot LTL checker from conda-forge.
- **rvln-server** (`rvln-server_env.yml`): PyTorch + transformers + flash-attn for OpenVLA inference. Requires a CUDA GPU.

Install the `rvln` package in both:

```bash
conda activate rvln-sim    && pip install -e .
conda activate rvln-server && pip install -e ".[server]"
```

Note: the `spot` LTL model checker is only available via conda-forge (not pip), so it lives in `rvln-sim_env.yml` and not in `pyproject.toml`.

API keys are read from `.env` first (shared defaults) and then `.env.local` (per-machine overrides). Put secrets only in `.env.local`. See `rvln.paths.load_env_vars` for the exact load order.

## Task JSON Format

Tasks live under `tasks/{system,ltl,goal_adherence,uav_flow,uav_flow_instructions}/`.

```json
{
  "instruction": "Go to the red building then land near the tree",
  "initial_pos": [-600, -1270, 128, 0, 61],
  "max_steps_per_subgoal": 200,
  "diary_check_interval": 10,
  "max_corrections": 5
}
```

`initial_pos` is `[x, y, z, pitch, yaw]` in Unreal coordinates. `diary_check_interval` controls how often (in steps) the diary monitor is invoked, and `max_corrections` caps how many corrective commands it may issue per subgoal.

## Common Issues

- **`modelscope` not found when downloading the simulator**: `pip install modelscope` inside `rvln-sim` before re-running `tools/download_simulator.py`.
- **`flash-attn` build fails**: it needs CUDA 11.6+ with matching toolkit. Install the matching wheel manually, or edit the server to use a different `attn_implementation`.
- **OpenVLA server exits on startup**: check `weights/OpenVLA-UAV/config.json` exists. `start_server.py` auto-descends into a single checkpoint subdir; if you have multiple, pass `--model-dir` explicitly.
- **`UnrealEnv not set` or env not found**: `export UnrealEnv=$(pwd)/runtime/unreal` or verify the downloader finished without errors.
- **Integration run hangs at startup**: confirm the server is reachable on port 5007 (`curl localhost:5007/health`) and that the Unreal binary is executable (`chmod +x` is applied by the downloader on Linux).

## Vendored Code

`src/gym_unrealcv/`, `src/rvln/eval/`, and `src/rvln/server/` contain code vendored from [UAV-Flow](https://github.com/buaa-colalab/UAV-Flow) (commit `0114801`). Upstream licensing applies to those subtrees; see the original repository for attribution requirements before redistributing.
