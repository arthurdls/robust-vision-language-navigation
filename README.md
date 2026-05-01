# Robust Vision-Language Navigation

Neuro-symbolic LTL planner with vision-language supervision for UAV control in simulation and on real hardware (MiniNav).

This system decomposes multi-step natural language instructions into an LTL formula, converts each subgoal into short OpenVLA commands, and supervises execution with a goal adherence VLM monitor that can issue corrections, detect stalls, and request operator help. It runs in both an Unreal Engine simulation (via UnrealCV) and on real drones (via MiniNav TCP).

## Architecture

```
Instruction: "Go to the red building, then land near the tree, never fly over the highway"
    |
    v
LTL Planner (LLM -> Spot automaton -> constraint classification)
    |
    v  subgoal: "approach the red building"
    |  constraints: [AVOID: "Flying over the highway"]
SubgoalConverter (LLM -> short OpenVLA command + OOD check)
    |
    |  if outside distribution -> ask_help / stop_reason="ood"
    v  command: "fly toward building"
OpenVLA Server (VLA model, returns drone actions)
    |
    v  action: [dx, dy, dz, dyaw]
Unreal Sim / MiniNav Hardware
    ^                       |
    |  periodic frames      |  convergence (drone stops)
    v                       v
GoalAdherenceMonitor -----> Supervisor Mode
  |  checkpoint every       |  evaluate: complete / stopped short / overshot
  |  N steps (frame-based)  |  issue corrective command to OpenVLA
  |  or N seconds (async)   |  if budget exhausted -> ask_help
  |                         |
  |  stall detection:       |
  |  completion plateau     |
  |  over last K checks     |
  |  -> ask_help            |
  v                         v
Operator Help (hardware) or stop_reason="ask_help" (simulation)
  [1] New low-level instruction
  [2] Replan from new high-level instruction
  [3] Skip subgoal
  [4] Abort mission
```

### Temporal Constraints

The LTL planner automatically classifies predicates as goals or constraints using BDD queries on the Spot automaton. Constraints carry a polarity:

- **Negative (avoidance)**: `G(!pi_X)` (never do X) or `!pi_X U pi_Y` (avoid X until Y). The VLM prompt labels these as `AVOID: ...`.
- **Positive (maintenance)**: `G(pi_X)` (always maintain X) or `pi_X U pi_Y` (maintain X until Y). The VLM prompt labels these as `MAINTAIN: ...`.

Active constraints are passed to the GoalAdherenceMonitor, which injects them into VLM prompts. On violation, the monitor triggers `force_converge` and the supervisor issues a corrective command (e.g., "move away from building B" or "ascend to restore altitude"), sharing the same correction budget as normal convergence corrections.

### Prompt Separation (With vs. Without Constraints)

The goal adherence monitor maintains **separate prompt templates** for subgoals that have active constraints and subgoals that do not (see `src/rvln/ai/prompts.py`). For example, `DIARY_GLOBAL_PROMPT` is used when there are no constraints, while `DIARY_GLOBAL_PROMPT_WITH_CONSTRAINTS` is used when constraints are present. The same pattern applies to the convergence prompts.

This separation is intentional: when no constraints are active, the VLM prompt contains no mention of constraints, the `constraint_violated` JSON field, or constraint-related instructions. Including constraint language in a prompt where no constraints exist would pollute the LLM's context with irrelevant concepts, potentially biasing its responses (e.g., hallucinating constraint violations, or spending reasoning capacity on an inapplicable field). The monitor selects the appropriate template at runtime based on whether `self._constraints` is non-empty (see `_format_global_prompt` and `_format_convergence_prompt` in `goal_adherence_monitor.py`).

When adding new prompt variants, follow this pattern: create a constraint-free version and a separate `_WITH_CONSTRAINTS` version rather than conditionally injecting constraint blocks into a single template.

### Checkpoint Modes

The GoalAdherenceMonitor supports two checkpoint modes:

- **Frame-based (default)**: `on_frame()` runs a VLM checkpoint every `diary_check_interval` steps. Used by `run_integration.py` (simulation). Synchronous: the control loop blocks during VLM calls.
- **Time-based (async)**: When `check_interval_s` is set, a background thread runs VLM checkpoints every N seconds. The control loop continues sending commands concurrently. Used by `run_hardware.py` (MiniNav) where blocking on VLM calls would stall the command stream.

### Operator Help

When the monitor detects a stall (completion plateau over the last `stall_window` checkpoints), exhausts its correction budget, hits the step/time limit, or the subgoal converter flags the instruction as outside OpenVLA's training distribution:

- **Hardware (`run_hardware.py`)**: The drone pauses and the operator is prompted with four options: new instruction, replan, skip, or abort.
- **Simulation (`run_integration.py`)**: The subgoal ends with `stop_reason="ask_help"` (or `stop_reason="ood"` for out-of-distribution) in the run summary. No interactive prompt (simulation runs are typically unattended).

Stall detection is controlled by `--stall_window` (default 3), `--stall_threshold` (default 0.05), and `--stall_completion_floor` (default 0.8).

### Out-of-Distribution Detection

The SubgoalConverter uses the LLM to assess whether a converted instruction falls outside OpenVLA's training distribution. OpenVLA was fine-tuned on first-person drone navigation in outdoor/suburban environments. Instructions involving indoor manipulation (pick up, grasp), non-drone locomotion (walk, drive), objects absent from typical drone footage (kitchen appliances, office furniture), or non-navigation tasks (answer a question, take a photo) are flagged as out-of-distribution. When an OOD instruction is detected:

- **Hardware**: The same operator help prompt is triggered immediately, before any actions are sent. The operator can provide a replacement instruction, replan, skip, or abort.
- **Simulation**: The subgoal returns immediately with `stop_reason="ood"` and zero steps executed.

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
    config.py             Centralized model and numeric defaults
    paths.py              Centralized path constants and env loading
    ai/                   LTL planner, goal adherence monitor, subgoal converter, LLM providers
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
| `scripts/run_integration.py` | Full LTL planner + goal adherence monitor pipeline |
| `scripts/run_ltl.py` | LTL-only control loop (no goal adherence supervision) |
| `scripts/run_goal_adherence.py` | Single-subgoal goal adherence experiments |
| `scripts/run_repl.py` | Interactive REPL for drone commands |
| `scripts/playback.py` | FPV viewer and MP4 encoder for saved runs |
| `scripts/scout_locations.py` | Position scouting helper for task authoring |
| `scripts/start_mock_hardware.py` | Simulated MiniNav drone-side hardware (TCP control sink + HTTP frame feed) |
| `scripts/run_hardware.py` | MiniNav real-drone integration pipeline |

## Running on Hardware (MiniNav)

The same planner + goal adherence monitor + OpenVLA stack can drive a real drone via the MiniNav interface in `src/rvln/mininav/`. The drone-facing module streams commands `[frame_count, vx, vy, vz, yaw]` as `float32` over TCP to a control server on the vehicle.

### What you need

- A flight controller / companion computer running a TCP control server that accepts the 5-float packet format (see `src/rvln/mininav/mock_server.py` for the wire format and a local simulator you can develop against).
- A USB / network camera visible to the machine running `rvln-sim`, or `--camera_url` pointed at any HTTP JPEG/PNG endpoint (the simulated hardware exposes one on `:8081/frame`).
- The OpenVLA server reachable on the network (typically on the same laptop that runs the pipeline).
- Optional but recommended: an external odometry source (HTTP poll or UDP JSON stream). Without one, the runner falls back to dead-reckoning from commanded velocities.

### Dry run against the simulated hardware

`scripts/start_mock_hardware.py` stands in for both halves of the drone-side companion: the TCP control sink (port `--port`, default 8080) and an HTTP frame feed (port `--frame_port`, default 8081) that serves random PNGs auto-discovered from `results/**/frames/`. If no frames are found it falls back to a generated white JPEG, so the pipeline runs end-to-end with no real camera attached.

```bash
# Terminal 1: OpenVLA server (GPU machine)
conda activate rvln-server
python scripts/start_server.py

# Terminal 2: simulated drone-side hardware (TCP control + HTTP frame feed)
conda activate rvln-sim
python scripts/start_mock_hardware.py --host 127.0.0.1 --port 8080 --frame_port 8081

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

### Operator help

When the monitor detects a stall, exhausts its correction budget, hits the step/time limit, or the subgoal is flagged as outside OpenVLA's distribution, the drone pauses and the operator is prompted:

```
[1] New low-level instruction   - replace the OpenVLA command, stay in current subgoal
[2] Replan from high-level      - re-run LTL planner with a new mission instruction
[3] Skip                        - continue or end the current subgoal
[4] Abort                       - stop the mission
```

Stall detection is controlled by `--stall_window` (default 3 checkpoints), `--stall_threshold` (default 0.05), and `--stall_completion_floor` (default 0.8). A time budget can be set with `--max_seconds_per_subgoal`.

### Safety checklist before arming

- Test the full pipeline against `rvln.mininav.mock_server` first and inspect the CSV.
- Cap `--max_steps_per_subgoal`, `--max_corrections`, and `--max_seconds_per_subgoal` to conservative values for the first flight.
- Keep a manual override channel on the drone; the pipeline will send commands at `--command_dt_s` (default 0.1 s) until a subgoal completes, a `stop` action is issued, the operator intervenes, or you Ctrl-C.

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

## Running Across Multiple PCs

The system has three main processes that can each run on a different machine:

1. **Unreal Simulator** (renders the environment, listens on UnrealCV port)
2. **OpenVLA Server** (GPU inference, Flask on port 5007)
3. **Pipeline Client** (LTL planner, goal adherence monitor, sends commands)

By default everything runs on `127.0.0.1`. To split across machines, set the networking environment variables in `.env.local` on the **pipeline client** machine (the one running `run_integration.py` or similar):

```bash
# .env.local on the pipeline client
export SERVER_HOST="10.0.0.2"   # IP of the machine running start_server.py
export SIM_HOST="10.0.0.3"      # IP of the machine running the Unreal simulator
export SIM_PORT="9000"           # UnrealCV port (only change if non-default)
```

These are read by `rvln.config` at startup and fed into the gym environment and OpenVLA client automatically.

### Example: two-machine setup

A common split is GPU machine + workstation, where the GPU machine hosts both the OpenVLA server and the Unreal sim, and the workstation runs the pipeline.

| Machine | Runs | IP (example) |
|---------|------|--------------|
| GPU box | `start_server.py`, Unreal binary | `10.0.0.2` |
| Workstation | `run_integration.py` | `10.0.0.5` |

On the **GPU box**:

```bash
# Terminal 1: OpenVLA server (binds to 0.0.0.0 so it's reachable remotely)
conda activate rvln-server
python scripts/start_server.py --host 0.0.0.0

# Terminal 2: Unreal simulator
# The simulator binary listens on SIM_PORT (default 9000).
# No extra flags needed; UnrealCV already binds to 0.0.0.0.
```

On the **workstation**:

```bash
# .env.local
export OPENAI_API_KEY="sk-..."
export SERVER_HOST="10.0.0.2"
export SIM_HOST="10.0.0.2"
```

```bash
conda activate rvln-sim
python scripts/run_integration.py --task first_task.json
```

### Example: three-machine setup

Separate the simulator onto its own machine when GPU memory is tight or when you want to isolate rendering from inference.

| Machine | Runs | IP (example) |
|---------|------|--------------|
| GPU box | `start_server.py` | `10.0.0.2` |
| Render box | Unreal binary | `10.0.0.3` |
| Workstation | `run_integration.py` | `10.0.0.5` |

The workstation's `.env.local`:

```bash
export SERVER_HOST="10.0.0.2"
export SIM_HOST="10.0.0.3"
```

### Networking checklist

- The OpenVLA server must be started with `--host 0.0.0.0` to accept remote connections (it defaults to `127.0.0.1`).
- Verify the server is reachable: `curl http://<SERVER_HOST>:5007/health` from the client machine.
- Verify the simulator is reachable: the UnrealCV port (`SIM_PORT`, default 9000) must not be blocked by a firewall.
- API keys (OpenAI, Gemini) are only needed on the pipeline client, not on the GPU or render machines.
- The `UnrealEnv` path is only relevant on the machine that actually runs the Unreal binary. The pipeline client does not need the simulator files if `SIM_HOST` points elsewhere, but `env_setup.py` will still set the default, so either leave it unset or point it at a dummy path.

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

`initial_pos` is `[x, y, z, pitch, yaw]` in Unreal coordinates. `diary_check_interval` controls how often (in steps) the goal adherence monitor is invoked, and `max_corrections` caps how many corrective commands it may issue per subgoal before asking the operator for help.

## Common Issues

- **`modelscope` not found when downloading the simulator**: `pip install modelscope` inside `rvln-sim` before re-running `tools/download_simulator.py`.
- **`flash-attn` build fails**: it needs CUDA 11.6+ with matching toolkit. Install the matching wheel manually, or edit the server to use a different `attn_implementation`.
- **OpenVLA server exits on startup**: check `weights/OpenVLA-UAV/config.json` exists. `start_server.py` auto-descends into a single checkpoint subdir; if you have multiple, pass `--model-dir` explicitly.
- **`UnrealEnv not set` or env not found**: `export UnrealEnv=$(pwd)/runtime/unreal` or verify the downloader finished without errors.
- **Integration run hangs at startup**: confirm the server is reachable on port 5007 (`curl localhost:5007/health`) and that the Unreal binary is executable (`chmod +x` is applied by the downloader on Linux).

## Vendored Code

`src/gym_unrealcv/`, `src/rvln/eval/`, and `src/rvln/server/` contain code vendored from [UAV-Flow](https://github.com/buaa-colalab/UAV-Flow) (commit `0114801`). Upstream licensing applies to those subtrees; see the original repository for attribution requirements before redistributing.
