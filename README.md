# Closing the Language-Level Loop: LTL Planning with Online Goal Adherence Monitoring for UAV Navigation

<p align="center">
  <strong>Arthur De Los Santos</strong>&nbsp;&nbsp;&nbsp;
  <strong>Makram Chahine</strong>&nbsp;&nbsp;&nbsp;
  <strong>Wei Xiao</strong>&nbsp;&nbsp;&nbsp;
  <strong>Daniela Rus</strong>
</p>

<p align="center">
  MIT CSAIL, Distributed Robotics Lab&nbsp;&nbsp;|&nbsp;&nbsp;Worcester Polytechnic Institute
</p>

<p align="center">
  <a href="https://github.com/arthurdls/robust-vision-language-navigation"><img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="GitHub"></a>&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/Status-Preprint-orange" alt="Preprint">
</p>

---

<p align="center">
  <img src="system_diagram.png" alt="RVLN System Overview" width="90%">
</p>

<p align="center"><em>
<b>RVLN</b> compiles free-form instructions into an LTL-NL automaton (left), executes each sub-goal with a frozen OpenVLA controller (center), and supervises progress with a VLM-based GoalAdherenceMonitor that maintains a running text diary over local and global image grids (right). On premature convergence, the supervisor issues corrective imperatives back to the controller. Every signal exchanged between the supervisors and the controller is a natural-language string, so the loop is policy-agnostic, fully auditable, and can drive systems built entirely from foundation models.
</em></p>

## Abstract

Vision-Language-Action (VLA) policies have enabled language-conditioned control for UAVs, but they fail on long-horizon missions: they drift from multi-step instructions, hallucinate sub-goal completion before visual evidence supports it, and offer no introspective signal for detecting or correcting failures. We introduce **RVLN**, a hierarchical neuro-symbolic system that closes the loop *at the language level*. An automatic NL-to-LTL-NL compiler translates free-form instructions into a deterministic monitor automaton with human-readable predicates, while a **GoalAdherenceMonitor** periodically queries a vision-language model over local two-frame and global nine-frame image grids, maintaining a running natural-language diary that gives the VLM short-term memory across checkpoints. RVLN removes the hand-authored LTL assumption of prior neuro-symbolic planners, detects and corrects premature VLA convergence via zero-shot VLM supervision, produces a human-readable audit log of every supervisory decision, and escalates to operators when progress plateaus. On a six-sub-goal urban UAV mission across seven ablation conditions, the full system is the only one to complete the task. We also demonstrate the language-level loop on real drone hardware: with the VLA replaced by a VLM controller emitting discrete drive actions, RVLN drives a real drone through all four sub-goals of an outdoor mission, suggesting the loop is policy-agnostic.

## Key Results

**Preliminary** single-run results (n=1 per condition) on a six-sub-goal mission in UnrealZoo Downtown West:

> *"Go past the first streetlamp, then turn until you see the black car, then go to the black car, then turn toward the red car and go to the red car. Finally, land in front of the traffic light pole on the right."*

| | Condition | Sub-goals | Task Success | Steps | VLM Calls | Corr. | Time (s) | Stop Reason |
|:---:|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **C0** | **Full System (RVLN)** | **6/6** | **✓** | 650 | 146 | 10 | 868 | monitor_complete |
| C1 | Naive VLA | 0/6 | ✗ | 56 | 0 | 0 | 29 | convergence |
| C2 | LLM Planner | 5/6 | ✗ | 758 | 179 | 25 | 1129 | ask_help (corr.) |
| C3 | Open-Loop LTL | 1/6 | ✗ | 281 | 0 | 0 | 152 | auto-advance |
| C4 | Single Frame | 3/6 | ✗ | 1180 | 149 | 27 | 1093 | ask_help (steps) |
| C5 | Grid Only | 2/6 | ✗ | 949 | 141 | 42 | 1340 | ask_help (corr.) |
| C6 | Text-Only Global | 1/6 | ✗ | 919 | 224 | 36 | 1062 | ask_help (corr.) |

With n=1 per condition, hypothesis tests and significance language are inappropriate and we avoid them. What the table establishes is a qualitative pattern: only the full language-level loop completes the mission, and every component the ablations remove shows up as a different abort pattern on the same hard sub-goal.

**Key findings:**
- **Decomposition is necessary.** C1 (Naive) collapses to the first action and stops after 56 steps with 0/6 sub-goals.
- **Monitoring is necessary.** C3 (Open-Loop LTL), with the same decomposition as C0, false-converges through every sub-goal in 281 total steps and reaches only 1/6.
- **The diary helps over single-frame checks.** C4 reaches 3/6 while consuming 1.8x more steps and 2.7x more corrections than C0.
- **Visual grids matter.** C6 (text-only) reaches 1/6: a text-only LLM cannot visually confirm the drone has arrived at the intended target rather than a similar one.
- **LTL formalism vs. flat list.** On a strictly sequential mission C2 also decomposes into all six sub-goals; the LTL chain and the flat list are formally similar here. The expressiveness advantage of LTL (OR branches, concurrent obligations) requires non-sequential structures left to future work.

### Hardware demonstration

A complementary hardware run drove a custom MiniNav quadcopter through a four-sub-goal outdoor mission with a VLM (gpt-5.4) replacing the OpenVLA controller, emitting discrete drive actions every ~3 seconds. Sub-goals 1, 2, and 3 each completed via `monitor_complete` at peak completion 1.00 with zero corrective re-prompts. The operator ended the run for time during sub-goal 4, with the drone in continuous forward progress toward the cardboard cutout (peak completion 0.47).

This is preliminary single-mission evidence that the language-level loop is policy-agnostic: a VLM can stand in for the VLA without any change to the supervisory machinery, because everything the supervisors emit is already a natural-language string the VLM can act on directly.

## Demos

### Hardware: outdoor four-sub-goal mission (2x speed)

<video src="https://github.com/arthurdls/robust-vision-language-navigation/raw/main/results/hardware/replay_synced_2x_compressed.mp4" controls width="100%"></video>

[Watch on GitHub if the embed does not render](https://github.com/arthurdls/robust-vision-language-navigation/raw/main/results/hardware/replay_synced_2x_compressed.mp4) (1x version: [`replay_synced_compressed.mp4`](https://github.com/arthurdls/robust-vision-language-navigation/raw/main/results/hardware/replay_synced_compressed.mp4)).

### Simulation: full system completing all six sub-goals (C0)

<video src="https://github.com/arthurdls/robust-vision-language-navigation/raw/main/results/condition0/downtown_west/c0_full_system__streetlamp_to_cars__2026_05_07_03_16_45/playback.mp4" controls width="100%"></video>

[Watch on GitHub if the embed does not render](https://github.com/arthurdls/robust-vision-language-navigation/raw/main/results/condition0/downtown_west/c0_full_system__streetlamp_to_cars__2026_05_07_03_16_45/playback.mp4). Per-condition recordings for C1-C6 are under `results/condition{N}/`.

## Contributions

1. **RVLN**, a neuro-symbolic system that combines automatic LTL-NL compilation with online VLM-based goal-adherence monitoring over a frozen VLA, structured around a language-level loop independent of the underlying policy.
2. **GoalAdherenceMonitor**: a zero-shot, prompt-only supervisor that maintains a natural-language diary annotated with past-corrective markers, plus a peak-dropoff override that advances the automaton when the per-sub-goal peak completion has reached 0.9 and the latest estimate has dropped 0.25 below that peak.
3. **A seven-condition ablation** on a six-sub-goal urban UAV mission. C0 completes the task; no baseline does.
4. **A hardware demonstration** with the VLA surgically removed and gpt-5.4 emitting discrete drive actions, driving a real drone through all four sub-goals of an outdoor mission before the operator ended the run for time during the fourth.
5. **Full codebase release** including the LTL planner, goal adherence monitor, hardware interface, and the run records that back every claim above (see `results/README.md` for the file-to-claim mapping).

## Method Overview

RVLN operates as a five-stage pipeline at two timescales:

1. **NL-to-LTL-NL Compilation.** A few-shot LLM call lifts free-form English into an LTL-NL formula with natural-language atomic predicates, and the formula is compiled into a deterministic monitor automaton via a standard LTL-to-automaton tool (we use [Spot](https://spot.lre.epita.fr/)).

2. **SubgoalConverter.** Each LTL-NL sub-goal is translated into a clean OpenVLA imperative by stripping visual stopping conditions (e.g., "turn right *until you see the red car*" becomes "turn right"). The GoalAdherenceMonitor enforces the original stopping condition externally.

3. **OpenVLA Execution.** The frozen 7B-parameter VLA controller (Flow-tuned for UAV) generates actions from egocentric RGB frames and the converted imperative.

4. **GoalAdherenceMonitor.** At each checkpoint two VLM queries run:
   - *Local query*: a two-frame grid (previous + current) for change detection. The one-sentence response is appended to a running text diary.
   - *Global query*: a nine-frame grid sampled across the trajectory, sent together with the diary, displacement vector, and sub-goal text, returns a JSON completion assessment.
   - *Stall detection*: when completion plateaus over the last 10 checkpoints below the 0.5 floor, the monitor escalates to the operator.

5. **Supervisor Mode.** On convergence (the drone stops moving) or a forced convergence, the VLM evaluates completion. If incomplete, a single-action corrective imperative (e.g. "Move closer to the black car") is issued back to the controller. Every issued correction is recorded as a `[CONVERGENCE @ step N]: corrective issued (<diagnosis>, <pct>% complete) -- "<instruction>"` marker in the running diary. The diary preface instructs the VLM to switch axes (e.g., altitude or a different turn direction) when completion has not improved since the most recent marker, instead of reissuing the same command. A complementary **peak-dropoff override** advances to the next sub-goal whenever the per-sub-goal peak completion estimate has previously reached 0.9 and the latest estimate has dropped at least 0.25 below that peak: a peak that high means the goal was at some point near-complete (so it has probably been achieved), and a 0.25+ retreat means the agent has moved away and further correction is unlikely to help.

### Foundation models used in the paper

The paper holds the foundation models fixed across every condition: `gpt-4o` handles all text-only planning (the NL-to-LTL-NL compiler, the SubgoalConverter, and the C2 flat-list decomposer), and `gpt-5.4` handles every VLM call in the GoalAdherenceMonitor (local two-frame, global nine-frame, convergence, and the text-only variant in C6).

## Getting Started

### Prerequisites

- **CUDA GPU(s)** for the OpenVLA server and the Unreal simulator. The paper runs used an RTX 5000 Ada for the OpenVLA server and an RTX 3060 for the simulator; a single GPU with comparable memory works too.
- **conda** (Miniconda or Anaconda)
- **~20 GB disk** for model weights + Unreal environment
- **OpenAI API key** for the LLM / VLM calls (paper uses `gpt-4o` for planning and `gpt-5.4` for the GoalAdherenceMonitor)

### Installation

```bash
# Clone
git clone git@github.com:arthurdls/robust-vision-language-navigation.git
cd robust-vision-language-navigation

# Create both conda envs and scaffold .env.local
bash tools/setup.sh

# Install the rvln package in each env
conda activate rvln-sim    && pip install -e .
conda activate rvln-server && pip install -e ".[server]"

# Configure API keys
$EDITOR .env.local          # OPENAI_API_KEY

# Download assets (~20 GB)
conda activate rvln-sim
python tools/download_weights.py       # -> weights/OpenVLA-UAV/
python tools/download_simulator.py     # -> runtime/unreal/
```

### Running

```bash
# Terminal 1: Start the OpenVLA server (GPU required)
conda activate rvln-server
python scripts/start_server.py

# Terminal 2: Run the full pipeline on the paper task
conda activate rvln-sim
python scripts/run_integration.py --task streetlamp_to_cars
```

Most commands are also wrapped in the `Makefile` (`make setup`, `make download-weights`, `make server`, `make run`).

### Reproducing the Paper

The paper reports a single n=1 run per condition on the six-sub-goal `streetlamp_to_cars` task in UnrealZoo Downtown West.

```bash
# Terminal 1: OpenVLA server
conda activate rvln-server
python scripts/start_server.py

# Terminal 2: All seven conditions
conda activate rvln-sim
python scripts/run_all_conditions.py --map downtown_west --task streetlamp_to_cars
```

To run a subset:

```bash
# Specific conditions on the paper task
python scripts/run_all_conditions.py --map downtown_west --conditions 0,2,4

# A different task or map
python scripts/run_all_conditions.py --map greek_island
```

Results are written to `results/condition<N>/<map_dir>/`. Completed tasks are tracked via `run_info.json`, so interrupted runs can be resumed by re-running the same command (already-completed tasks are skipped, aborted/crashed tasks are retried). The committed `results/` directory contains the exact run records that back every number in the paper; see `results/README.md`.

## Repository Structure

```
robust-vision-language-navigation/
  src/rvln/                 Core Python package
    ai/                     LTL planner, GoalAdherenceMonitor, SubgoalConverter, LLM providers
    sim/                    Unreal sim environment setup, pose utilities
    eval/                   Batch evaluation runner, metrics, playback
    server/                 OpenVLA inference server
    mininav/                MiniNav real-drone interface (TCP control + camera + odometry)
  src/gym_unrealcv/         UnrealCV gym environments (vendored from UAV-Flow)
  scripts/                  CLI entry points
    run_all_conditions.py   Orchestrator: runs conditions across maps
    start_sim_controller.py Remote simulator lifecycle daemon
    run_integration.py      Full system (C0)
    run_condition{2-6}_*.py Ablation conditions
    start_simulator.py      Launches Unreal binary + sim API server
    start_server.py         OpenVLA server
    run_hardware_openvla.py Real-drone pipeline (OpenVLA driver)
    run_hardware_gpt.py     Real-drone pipeline (GPT-5.4 driver, used for the hardware demo)
    run_repl.py             Interactive drone REPL
    playback.py             FPV viewer and MP4 encoder
  tasks/                    Task JSON definitions
  tools/                    Setup, weight/simulator download scripts
  results/                  Run records backing the paper claims (see results/README.md)
  tests/                    Test suite
```

## Hardware Deployment (MiniNav)

The same pipeline runs on real drones via the MiniNav interface. The drone-facing module streams commands `[frame_count, vx, vy, vz, yaw]` as `float32` over TCP.

```bash
# Dry run against simulated hardware
python scripts/start_mock_hardware.py --host 127.0.0.1 --port 8080 --frame_port 8081

# Live flight: GPT-5.4-driven controller (the hardware demo in the paper)
python scripts/run_hardware_gpt.py \
  --preferred_server_host 192.168.0.101 \
  --control_port 8080 \
  --camera 0 \
  --diary_mode time \
  --diary_check_interval_s 3.0

# Live flight: OpenVLA-driven controller
python scripts/run_hardware_openvla.py \
  --preferred_server_host 192.168.0.101 \
  --control_port 8080 \
  --camera 0 \
  --initial_position 0,0,0,0 \
  --odom_udp_port 9001 \
  --command_is_velocity
```

The hardware interface supports interactive operator help (new instruction, replan, skip, abort), time-based asynchronous diary checkpoints, and external odometry via HTTP or UDP.

**Pipelined VLM checkpoints.** With `--diary-mode time` and `--diary_check_interval_s` (e.g. 3 s), the goal-adherence monitor dispatches VLM calls concurrently and writes `checkpoint_NNNN/` directories in strict dispatch order. Out-of-order returns from the LLM provider are reordered before publish so the dashboard never visually regresses; hung calls are skipped after 30 s. The pool size and per-step timeout are configurable via `--monitor_max_inflight` and `--monitor_dispatch_timeout_s`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{delossantos2026rvln,
  title  = {Closing the Language-Level Loop: {LTL} Planning with Online Goal Adherence
            Monitoring for {UAV} Navigation},
  author = {De Los Santos, Arthur and Chahine, Makram and Xiao, Wei and Rus, Daniela},
  year   = {2026}
}
```

## Acknowledgments

This work was supported by the MIT Generative AI Impact Consortium (MGAIC) and by The Boeing Company, administered through the Distributed Robotics Lab at MIT CSAIL.

## License

`src/gym_unrealcv/`, `src/rvln/eval/`, and `src/rvln/server/` contain code vendored from [UAV-Flow](https://github.com/buaa-colalab/UAV-Flow) (commit `0114801`). Upstream licensing applies to those subtrees; see the original repository for attribution requirements.

## Ethical Use Disclaimer

This research was developed for civilian scientific purposes: advancing autonomous UAV navigation through language grounding and formal verification. The authors do not endorse or support the use of this work, in whole or in part, for military operations, weapons development, surveillance of civilian populations, or any application intended to cause harm to human life.

We encourage all users and developers who build upon this work to consider the ethical implications of their applications and to prioritize human safety and well-being.
