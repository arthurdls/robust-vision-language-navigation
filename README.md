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
| `scripts/start_simulate_hardware.py` | Simulated MiniNav drone-side hardware (TCP control sink + HTTP frame feed) |
| `scripts/run_hardware.py` | MiniNav real-drone integration pipeline |

## Running on Hardware (MiniNav)

The same planner + diary monitor + OpenVLA stack can drive a real drone via the MiniNav interface in `src/rvln/mininav/`. The drone-facing module streams commands `[frame_count, vx, vy, vz, yaw]` as `float32` over TCP to a control server on the vehicle.

### What you need

- A flight controller / companion computer running a TCP control server that accepts the 5-float packet format (see `src/rvln/mininav/mock_server.py` for the wire format and a local simulator you can develop against).
- A USB / network camera visible to the machine running `rvln-sim`, or `--camera_url` pointed at any HTTP JPEG/PNG endpoint (the simulated hardware exposes one on `:8081/frame`).
- The OpenVLA server reachable on the network (typically on the same laptop that runs the pipeline).
- Optional but recommended: an external odometry source (HTTP poll or UDP JSON stream). Without one, the runner falls back to dead-reckoning from commanded velocities.

### Dry run against the simulated hardware

`scripts/start_simulate_hardware.py` stands in for both halves of the drone-side companion: the TCP control sink (port `--port`, default 8080) and an HTTP frame feed (port `--frame_port`, default 8081) that serves random PNGs auto-discovered from `results/**/frames/`. If no frames are found it falls back to a generated white JPEG, so the pipeline runs end-to-end with no real camera attached.

```bash
# Terminal 1: OpenVLA server (GPU machine)
conda activate rvln-server
python scripts/start_server.py

# Terminal 2: simulated drone-side hardware (TCP control + HTTP frame feed)
conda activate rvln-sim
python scripts/start_simulate_hardware.py --host 127.0.0.1 --port 8080 --frame_port 8081

# Terminal 3: Hardware pipeline pointed at the simulator
conda activate rvln-sim
python scripts/run_hardware.py \
  --preferred_server_host 127.0.0.1 \
  --control_port 8080 \
  --camera_url http://127.0.0.1:8081/frame \
  --openvla_predict_url http://127.0.0.1:5007/predict \
  --initial_position 0,0,0,0 \
  --command_is_velocity \
  --action_pose_mode delta_from_pose \
  --instruction "Move forward 10.0 meters, then turn toward the red car"
```

OpenVLA's proprio input and returned action pose are in **centimetres** (the server adds the raw displacement to the current position before returning, so the response is an absolute target in cm). That is why the simulated run pairs `--action_pose_mode delta_from_pose` (subtract current pose to recover the cm displacement) with `--command_is_velocity` (integrate that displacement as cm/s over `--command_dt_s`). Omitting either flag causes the dead-reckoned pose to compound each step instead of advancing linearly.

The mock writes every received command to a CSV in its working directory so you can verify the command stream before flying anything. Run results land under `results/hardware/run_<timestamp>/`.

### Live flight

```bash
conda activate rvln-sim
python scripts/run_hardware.py \
  --preferred_server_host 192.168.0.101 \
  --control_port 8080 \
  --openvla_predict_url http://<openvla-host>:5007/predict \
  --camera 0 \
  --initial_position 0,0,0,0 \
  --odom_udp_port 9001 \
  --command_is_velocity
```

`--preferred_server_host` is the drone's IP. If unreachable the runner falls back to the host's own LAN IP so you notice the problem quickly rather than silently streaming commands into the void.

Odometry options:

- `--odom_http_url http://<drone>:<port>/pose` for a poll endpoint returning `{"x", "y", "z", "yaw"}`.
- `--odom_udp_port 9001` to listen for UDP JSON packets with the same schema.
- Omit both and the runner dead-reckons from the commands it sends (`--command_is_velocity` tells it to treat commands as velocities integrated over `--command_dt_s` rather than as per-step positional deltas). Units follow whatever the flight controller expects; the bundled OpenVLA server emits poses in centimetres.

Action mapping:

- `--action_pose_mode direct` (default): pass the OpenVLA action pose through as the command payload.
- `--action_pose_mode delta_from_pose`: subtract the current relative pose first; useful when OpenVLA outputs absolute targets but the flight controller expects deltas.

Artifacts (frames, diary logs, trajectory, run_info.json) land in `results/hardware/run_<timestamp>/` (override with `--results_dir`) and have the same shape as the simulator runs, so `scripts/playback.py` and the offline goal-adherence tools work on them unchanged.

### Safety checklist before arming

- Test the full pipeline against `rvln.mininav.mock_server` first and inspect the CSV.
- Cap `--max_steps_per_subgoal` and `--max_corrections` to conservative values for the first flight.
- Keep a manual override channel on the drone; the pipeline will send commands at `--command_dt_s` (default 0.1 s) until a subgoal completes, a `stop` action is issued, or you Ctrl-C.

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
