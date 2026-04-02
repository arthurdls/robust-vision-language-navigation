# Goal Adherence Architecture & Experiment Results

This document describes the goal adherence pipeline — how an LLM-supervised
diary monitor turns a single natural-language subgoal into a closed-loop
execution cycle on top of the OpenVLA drone action model — and catalogues
every experiment run stored in `results/goal_adherence_results/`.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Pipeline Walkthrough](#3-pipeline-walkthrough)
4. [Module Reference](#4-module-reference)
5. [Coordinate Frames](#5-coordinate-frames)
6. [Task Format](#6-task-format)
7. [Experiment Results](#7-experiment-results)
8. [Cross-Experiment Analysis](#8-cross-experiment-analysis)

---

## 1. System Overview

The goal adherence system evaluates whether an LLM-based "diary monitor"
supervisor can improve a vision-language-action (VLA) drone model's ability to
complete a single natural-language subgoal. It runs two conditions per task:

- **Baseline (no_llm):** The raw subgoal string is sent directly to OpenVLA as
  the instruction. The drone flies until it stops moving (convergence) or
  reaches the step limit. No external monitoring or correction.

- **LLM-assisted (llm):** A `SubgoalConverter` rewrites the subgoal into a
  short imperative OpenVLA instruction. A `LiveDiaryMonitor` watches FPV frames
  every N steps, maintains a running diary of observations, tracks estimated
  completion percentage, and — when the drone stops prematurely — issues
  single-action corrective commands to resume progress. The drone flies under
  this closed-loop supervision until the monitor declares the subgoal complete,
  corrections are exhausted, or the step limit is reached.

### External dependencies

| Component | Role | How it runs |
|-----------|------|-------------|
| **OpenVLA server** | Predicts next-step action poses from (image, proprio, instruction) | `scripts/start_openvla_server.py` — Flask HTTP server on port 5007 |
| **Unreal Engine sim** | Renders FPV frames and moves the drone | `gym_unrealcv` DowntownWest environment via UAV-Flow-Eval |
| **VLM (GPT-4o)** | Powers the diary monitor and subgoal converter | OpenAI API via `LLMFactory` |

---

## 2. Architecture Diagram

```
                         GOAL ADHERENCE PIPELINE
 ┌─────────────────────────────────────────────────────────────────────┐
 │                                                                     │
 │   ┌─────────────┐                                                   │
 │   │  Task JSON   │  { subgoal, initial_pos, max_steps,              │
 │   │  (one file)  │    diary_check_interval }                        │
 │   └──────┬───────┘                                                  │
 │          │                                                          │
 │          ▼                                                          │
 │   ┌──────────────────┐    "Turn right until you see the red car"    │
 │   │ SubgoalConverter  │───────────────────────────────────────┐      │
 │   │ (GPT-4o, once)   │    converts to: "turn right"          │      │
 │   └──────────────────┘                                       │      │
 │                                                              │      │
 │   ┌──────────────────────────────────────────────────────────┼──┐   │
 │   │                   MAIN CONTROL LOOP                      │  │   │
 │   │                                                          │  │   │
 │   │   ┌────────────────┐    FPV image    ┌───────────────┐   │  │   │
 │   │   │  Unreal Engine ├───────────────►│  OpenVLA       │   │  │   │
 │   │   │  Simulator     │◄───────────────┤  Server        │   │  │   │
 │   │   │  (DowntownWest)│  action poses   │  (HTTP /predict│   │  │   │
 │   │   └───────┬────────┘                 │   port 5007)  │   │  │   │
 │   │           │                          └───────▲───────┘   │  │   │
 │   │           │ current_pose                     │           │  │   │
 │   │           │ + FPV frame          instruction │           │  │   │
 │   │           │                      + proprio   │           │  │   │
 │   │           │                                  │           │  │   │
 │   │   ┌───────▼──────────────────────────────────┴───────┐   │  │   │
 │   │   │              run_goal_adherence.py                │   │  │   │
 │   │   │                                                   │   │  │   │
 │   │   │  • Manages step loop and convergence detection    │   │  │   │
 │   │   │  • Translates coordinate frames                   │   │  │   │
 │   │   │  • Routes monitor decisions to OpenVLA            │   │  │   │
 │   │   └────────┬────────────────────────────▲─────────────┘   │  │   │
 │   │            │ every N steps              │ stop /           │  │   │
 │   │            │ + on convergence           │ continue /       │  │   │
 │   │            ▼                            │ command          │  │   │
 │   │   ┌────────────────────────────────────────────────────┐  │  │   │
 │   │   │            LiveDiaryMonitor (GPT-4o)               │  │  │   │
 │   │   │                                                    │  │  │   │
 │   │   │  PASSIVE MODE (every N steps):                     │  │  │   │
 │   │   │    1. Local VLM: 2-frame grid → "what changed?"    │  │  │   │
 │   │   │    2. Append observation to running diary          │  │  │   │
 │   │   │    3. Global VLM: 9-frame grid + diary →           │  │  │   │
 │   │   │       { complete, completion_%, on_track,           │  │  │   │
 │   │   │         should_stop }                              │  │  │   │
 │   │   │    → continue / force_converge / stop              │  │  │   │
 │   │   │                                                    │  │  │   │
 │   │   │  SUPERVISOR MODE (on convergence):                 │  │  │   │
 │   │   │    1. 9-frame grid + full diary → convergence VLM  │  │  │   │
 │   │   │       { complete, completion_%, diagnosis,          │  │  │   │
 │   │   │         corrective_instruction }                   │  │  │   │
 │   │   │    → "move forward" / "ascend 0.5 meters" /        │  │  │   │
 │   │   │      "continue turning right" / complete           │  │  │   │
 │   │   └────────────────────────────────────────────────────┘  │  │   │
 │   └───────────────────────────────────────────────────────────┘  │   │
 │                                                                  │   │
 │   Output per run:                                                │   │
 │     run_info.json, trajectory_log.json, diary_summary.json,      │   │
 │     diary_artifacts/{checkpoint_*, convergence_*}, playback.mp4  │   │
 └──────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Walkthrough

### 3.1 Initialization

1. **Load task JSON** from `tasks/goal_adherence_tasks/`. Validate required
   fields: `task_name`, `subgoal`, `initial_pos`, `max_steps`,
   `diary_check_interval`.

2. **Teleport drone** to `initial_pos` in the Unreal sim. Reset the OpenVLA
   server state (`POST /reset`).

3. **Subgoal conversion** (LLM runs only): `SubgoalConverter` sends the full
   subgoal to GPT-4o once and receives a short imperative instruction that
   OpenVLA can execute. Visual stopping conditions (e.g., "until you see X")
   are stripped; spatial proximity conditions are converted to approach commands.

   | Original subgoal | Converted instruction |
   |---|---|
   | "Turn right until you see the red car" | "turn right" |
   | "Continue forward until you are close to the person ahead" | "get closer to the person ahead" |
   | "Go above the pergola" | "go above the pergola" |
   | "Move through the pergola between the wooden poles" | "move through the pergola between the wooden poles" |

4. **Create `LiveDiaryMonitor`** (LLM runs only) with the original subgoal
   (not the converted instruction), check interval, VLM model, and artifacts
   directory.

### 3.2 Main loop (each step)

```
for step in range(max_steps):
    1. Capture FPV frame from the simulator
    2. [LLM] Call monitor.on_frame(frame, displacement)
       → stop: end run
       → force_converge: flag for convergence handling after this step
       → continue: proceed normally
    3. Compute OpenVLA proprio: current_pose - openvla_pose_origin
    4. POST to OpenVLA /predict with (image, proprio, instruction)
    5. Reframe predicted action_poses back to subtask coordinates
    6. Execute action_poses in the simulator → update current_pose
    7. Convergence detection (see §3.3)
    8. If converged: handle convergence (see §3.4)
```

### 3.3 Convergence detection

The drone is considered "converged" (stopped) when either:

- **Force converge:** The last diary checkpoint returned `force_converge`
  (the VLM detected completion or a problem like overshooting).
- **Pose stall:** For 10 consecutive steps, position deltas are < 3 cm on each
  axis and yaw delta is < 1 degree. This only triggers if at least
  `diary_check_interval` steps have passed since the last correction, preventing
  premature re-convergence.

### 3.4 Convergence handling

**Baseline (no_llm):** The first convergence ends the run (`stop_reason: "convergence"`).

**LLM-assisted:** The monitor's `on_convergence` method evaluates the situation:

1. Builds a 9-frame grid from sampled frames across the run.
2. Sends the grid + full diary + displacement to GPT-4o with the convergence
   prompt, requesting a JSON response:
   ```json
   {
     "complete": true/false,
     "completion_percentage": 0.0-1.0,
     "diagnosis": "stopped_short" | "overshot" | "complete",
     "corrective_instruction": "..." | null
   }
   ```
3. **If complete:** End run (`stop_reason: "llm_stopped"`).
4. **If not complete + corrective instruction provided:**
   - Replace `current_instruction` with the corrective command
   - Snapshot `current_pose` as the new `openvla_pose_origin` (so OpenVLA
     sees a fresh [0,0,0,0] start for the new instruction)
   - Reset the OpenVLA server (`POST /reset`)
   - Clear convergence counters and resume the main loop
5. **If corrections exhausted** (default max: 10): End run.

This correction cycle is the core mechanism that distinguishes LLM-assisted runs
from baselines — it allows the system to recover from premature stops and
incrementally refine the drone's behavior.

### 3.5 Output artifacts

| File | Condition | Contents |
|------|-----------|----------|
| `run_info.json` | Both | Task config, mode, total steps, stop reason, corrections used, completion %, timing |
| `trajectory_log.json` | Both | Per-step pose and action data |
| `diary_summary.json` | LLM only | Full diary text, override history, VLM call counts |
| `diary_artifacts/checkpoint_NNNN/` | LLM only | Local/global grids (PNG), prompts and responses (TXT), diary snapshot |
| `diary_artifacts/convergence_NNN/` | LLM only | Convergence prompt, response, diary snapshot |
| `frames/` | Both | Saved FPV frames (PNG) |
| `playback.mp4` | Optional | Encoded frame sequence video |

---

## 4. Module Reference

### 4.1 `scripts/run_goal_adherence.py`

The orchestrator script. Handles CLI arguments, simulation setup, the main
control loop, convergence detection, coordinate frame management, and result
serialization. Runs `RUNS_PER_CONDITION` (3) trials per condition per task.

### 4.2 `modules/subgoal_converter.py` — SubgoalConverter

A one-shot LLM call (temperature 0) that translates a natural-language subgoal
into a short imperative OpenVLA-compatible instruction. Called once at the start
of each LLM run. Rules strip visual detection conditions ("until you see X")
and convert proximity conditions into approach commands.

### 4.3 `modules/diary_monitor.py` — LiveDiaryMonitor

The central intelligence of the LLM-assisted condition. Operates in two modes:

**Passive monitoring (`on_frame`):** Every `check_interval` steps:
1. **Local VLM query:** Builds a 2-frame grid (start and end of interval),
   asks "what changed?" → one-sentence observation appended to the diary.
2. **Global VLM query:** Builds a grid of up to 9 sampled frames, sends the
   full diary + displacement + previous completion estimate, asks for a
   structured JSON assessment (`complete`, `completion_percentage`, `on_track`,
   `should_stop`).
3. Returns `continue`, `force_converge` (if complete or should_stop), or `stop`.

**Supervisor mode (`on_convergence`):** When the drone stops:
1. Builds a 9-frame grid, sends full context to the VLM.
2. Asks for diagnosis (`stopped_short`, `overshot`, `complete`) and a single
   corrective instruction.
3. Returns `stop` (if complete), `command` (with corrective instruction), or
   `stop` (if max corrections exhausted).

Key design principles from the system prompt:
- Never mark complete unless highly confident; cap at 0.95 when uncertain.
- Issue single-axis corrections (not compound "ascend and move forward").
- Keep corrections small (< 1.0 meters) for frequent re-evaluation.
- Retreat commands must reference the target object.

### 4.4 `modules/utils/vision_utils.py`

Utility functions shared across modules:
- `sample_frames_every_n`: Samples every N-th frame for temporal coverage.
- `build_frame_grid`: Composites frames into a single padded grid image.
- `query_vlm`: Provider-agnostic VLM call (image + text → text response).

### 4.5 `modules/utils/base_llm_providers.py`

`LLMFactory` with `OpenAIProvider` and `GeminiProvider`. Handles text-only,
multimodal (system + user + image), and text+image request patterns. Used by
both the diary monitor and subgoal converter.

### 4.6 `scripts/sim_common.py`

Shared simulation utilities:
- Coordinate transforms (`transform_to_global`, `relative_pose_to_world`)
- OpenVLA state formatting (`state_for_openvla` → 4-float `[x, y, z, yaw_deg]`)
- Sim environment bootstrapping and monkey-patches for gym_unrealcv
- `apply_action_poses`: Executes predicted poses in the simulator

### 4.7 `scripts/start_openvla_server.py`

Loads the fine-tuned OpenVLA-UAV model, serves Flask HTTP endpoints:
- `POST /predict`: Takes base64 image + proprio + instruction, returns action poses
- `POST /reset`: Resets model state (added on top of the original server)

---

## 5. Coordinate Frames

The system uses three coordinate frames that must stay aligned:

### 5.1 Subtask frame (diary monitor's reference)

Position `[x, y, z, yaw]` relative to the task's `initial_pos`. The diary
monitor always receives displacements in this frame so that tracking remains
consistent across the entire run, even through instruction changes.

- **x**: forward/backward relative to the initial heading
- **y**: left/right relative to the initial heading
- **z**: altitude change (cm)
- **yaw**: heading change (degrees)

The displacement shown to the VLM is converted to meters:
`[x/100, y/100, z/100, yaw°]`.

### 5.2 OpenVLA instruction frame

Position relative to the start of the *current instruction*, not the task.
When the monitor issues a corrective instruction:
1. `openvla_pose_origin` is set to the current subtask-frame pose.
2. Proprio sent to OpenVLA = `current_pose - openvla_pose_origin` → starts at `[0,0,0,0]`.
3. Predicted action poses are translated back: `action + openvla_pose_origin`
   to stay in the subtask frame before execution.

This prevents teleportation artifacts when switching instructions mid-run.

### 5.3 OpenVLA action output (body frame)

The OpenVLA model outputs actions in the **current body frame** of the drone:
- x = forward/backward relative to current heading
- y = left/right relative to current heading
- z = altitude, yaw = heading change (radians)

The server (`openvla_act.py`) rotates these back to the global frame using the
current yaw before returning them to the client.

---

## 6. Task Format

Tasks are stored as JSON in `tasks/goal_adherence_tasks/`. Required fields:

```json
{
  "task_name": "approach_person_ahead",
  "subgoal": "Continue forward until you are close to the person ahead",
  "initial_pos": [-10520, -1500, 170, 81],
  "max_steps": 300,
  "diary_check_interval": 10,
  "notes": "Start from further away"
}
```

| Field | Description |
|-------|-------------|
| `task_name` | Unique identifier, used as the results subdirectory name |
| `subgoal` | Natural-language goal — passed to the diary monitor and subgoal converter |
| `initial_pos` | `[x, y, z, yaw]` in Unreal world coordinates (cm, degrees) |
| `max_steps` | Step budget per run |
| `diary_check_interval` | Steps between diary monitor checkpoints |
| `notes` | Optional human-readable context |

---

## 7. Experiment Results

Six tasks were evaluated, each with 3 baseline runs (`no_llm_run_01-03`) and
up to 3 LLM-assisted runs (`llm_run_01-03`). All LLM runs used **GPT-4o**.
The executed step budget was **200** (the task JSONs specify 300, but runs were
launched with `max_steps=200`).

### 7.1 approach_person_ahead

**Goal:** "Continue forward until you are close to the person ahead"
**Converted instruction:** "move forward"
**Starting position:** Looking toward a person ~50m ahead at a low altitude.

#### LLM runs

| Run | Steps | Corrections | Completion | Outcome |
|-----|-------|-------------|------------|---------|
| llm_run_01 | 49 | 2 | 1.0 | Success |
| llm_run_02 | 89 | 4 | 1.0 | Success |
| llm_run_03 | 49 | 1 | 1.0 | Success |

All three runs reach full completion. Convergence diagnoses are consistently
`stopped_short` with corrective instructions of "move forward". Run 02 takes
longer due to multiple stops at a distance where the person is visible but not
yet close.

#### Baseline runs

| Run | Steps | Stop reason |
|-----|-------|-------------|
| no_llm_run_01 | 28 | convergence |
| no_llm_run_02 | 58 | convergence |
| no_llm_run_03 | 13 | convergence |

Baselines converge quickly — the raw instruction "Continue forward until you
are close to the person ahead" includes a visual stopping condition that
OpenVLA cannot act on, leading to early stops far from the goal.

---

### 7.2 approach_white_building

**Goal:** "Approach the white building"
**Converted instruction:** "approach the white building"
**Starting position:** Facing a white building from a distance.

#### LLM runs

| Run | Steps | Corrections | Completion | Outcome |
|-----|-------|-------------|------------|---------|
| llm_run_01 | 89 | 1 | 1.0 | Success |
| llm_run_02 | 79 | 0 | 1.0 | Success |
| llm_run_03 | 79 | 1 | 1.0 | Success |

The cleanest task — run 02 completes with zero corrections. When corrections
are needed, they are simple "move forward" nudges after `stopped_short` near
the building.

#### Baseline runs

| Run | Steps | Stop reason |
|-----|-------|-------------|
| no_llm_run_01 | 55 | convergence |
| no_llm_run_02 | 200 | max_steps |
| no_llm_run_03 | 82 | convergence |

Baselines show high variance. Run 02 drifts for the full 200 steps without
converging, while others stop at various distances.

---

### 7.3 go_above_pergola

**Goal:** "Go above the pergola"
**Converted instruction:** "go above the pergola"
**Starting position:** Near a pergola structure, below its top.

#### LLM runs

| Run | Steps | Corrections | Completion | Outcome |
|-----|-------|-------------|------------|---------|
| llm_run_01 | 130 | 1 | 1.0 | Success |
| llm_run_02 | 109 | 1 | 0.95 | Partial |
| llm_run_03 | 89 | 2 | 1.0 | Success |

This task most clearly demonstrates the value of **axis-specific corrections**.
OpenVLA moves forward toward the pergola but does not ascend sufficiently. The
monitor identifies the vertical gap and issues "ascend 0.5 meters" or "ascend
1.0 meters" corrections. Run 02 plateaus at 0.95 — the drone ascends once but
the monitor ends the run before full confirmation.

Example convergence response (run 01):
```json
{
  "complete": false,
  "completion_percentage": 0.55,
  "diagnosis": "stopped_short",
  "corrective_instruction": "ascend 0.5 meters"
}
```

#### Baseline runs

| Run | Steps | Stop reason |
|-----|-------|-------------|
| no_llm_run_01 | 100 | convergence |
| no_llm_run_02 | 105 | convergence |
| no_llm_run_03 | 101 | convergence |

Baselines consistently converge around step 100 at a position near but not
above the pergola — unable to self-correct the missing vertical component.

---

### 7.4 go_between_tree_and_streetlight

**Goal:** "Go between the tree and the streetlight"
**Converted instruction:** "go between the tree and the streetlight"
**Starting position:** Facing the gap between a tree and a streetlight.

#### LLM runs

| Run | Steps | Corrections | Completion | Outcome |
|-----|-------|-------------|------------|---------|
| llm_run_01 | 49 | 1 | 1.0 | Success |
| llm_run_02 | 40 | 0 | 1.0 | Success |
| llm_run_03 | 70 | 3 | 1.0 | Success |

Run 02 completes without any corrections — the checkpoint declares complete
and `force_converge` verification confirms it. Run 03 has the most correction
churn, with repeated `stopped_short` diagnoses at 0.85-0.95 and small "move
slightly forward" nudges.

#### Baseline runs

| Run | Steps | Stop reason |
|-----|-------|-------------|
| no_llm_run_01 | 89 | convergence |
| no_llm_run_02 | 81 | convergence |
| no_llm_run_03 | 73 | convergence |

Baselines converge consistently but without confirmation of goal achievement.

---

### 7.5 move_through_pergola

**Goal:** "Move through the pergola between the wooden poles"
**Converted instruction:** "move through the pergola between the wooden poles"
**Starting position:** Same as go_above_pergola — near the pergola.

#### LLM runs

| Run | Steps | Corrections | Completion | Outcome |
|-----|-------|-------------|------------|---------|
| llm_run_01 | 69 | 1 | 1.0 | Success |
| llm_run_02 | 60 | 0 | 1.0 | Success |
| llm_run_03 | 30 | 0 | 1.0 | Success |

Fastest task overall. Run 03 finishes in just 30 steps (3 checkpoints) with
zero corrections — the monitor's first checkpoint assessment triggers
`force_converge` with `complete: true`, and convergence verification confirms.
Run 01 needs one "move forward slightly through the pergola" correction.

#### Baseline runs

| Run | Steps | Stop reason |
|-----|-------|-------------|
| no_llm_run_01 | 46 | convergence |
| no_llm_run_02 | 200 | max_steps |
| no_llm_run_03 | 80 | convergence |

High variance in baselines — run 02 never converges within 200 steps.

---

### 7.6 turn_right_until_red_car

**Goal:** "Turn right until you see the red car"
**Converted instruction:** "turn right"
**Starting position:** Facing away from both the red and white cars.

**Note:** `llm_run_02` is missing from the results directory. Only 2 of 3 LLM
runs are available.

#### LLM runs

| Run | Steps | Corrections | Completion | Outcome |
|-----|-------|-------------|------------|---------|
| llm_run_01 | 199 | 9 | 0.95 | Partial |
| llm_run_03 | 199 | 9 | 1.0 | Success |

The hardest task in the suite. Both runs exhaust nearly the full 200-step
budget and use 9 corrections each — long sequences of `stopped_short` with
"continue turning right" instructions. The diary shows the drone gradually
rotating, with the red car eventually appearing at the edge and then center of
frame. Run 01 ends at 0.95 (the car is visible but the monitor is not fully
confident); run 03 reaches 1.0 at the final checkpoint.

Example convergence response (run 01, correction 8):
```json
{
  "complete": false,
  "completion_percentage": 0.80,
  "diagnosis": "stopped_short",
  "corrective_instruction": "continue turning right"
}
```

#### Baseline runs

| Run | Steps | Stop reason |
|-----|-------|-------------|
| no_llm_run_01 | 19 | convergence |
| no_llm_run_02 | 22 | convergence |
| no_llm_run_03 | 19 | convergence |

Baselines converge almost immediately (~20 steps). The raw instruction includes
the visual condition "until you see the red car" which OpenVLA cannot evaluate,
so it turns briefly and stops — far from actually seeing the red car.

---

## 8. Cross-Experiment Analysis

### 8.1 Summary table (LLM runs)

| Task | Runs | Avg steps | Avg corrections | Success rate |
|------|------|-----------|-----------------|--------------|
| approach_person_ahead | 3 | 62 | 2.3 | 3/3 (100%) |
| approach_white_building | 3 | 82 | 0.7 | 3/3 (100%) |
| go_above_pergola | 3 | 109 | 1.3 | 2/3 (67%) |
| go_between_tree_and_streetlight | 3 | 53 | 1.3 | 3/3 (100%) |
| move_through_pergola | 3 | 53 | 0.3 | 3/3 (100%) |
| turn_right_until_red_car | 2 | 199 | 9.0 | 1/2 (50%) |

### 8.2 Key observations

1. **LLM supervision adds goal verification and recovery.** Baseline runs
   converge based on pose stall — they have no notion of whether the goal
   was actually achieved. LLM runs terminate on confirmed completion (or
   explicitly flagged non-completion), providing much stronger guarantees.

2. **`stopped_short` dominates convergence diagnoses.** Nearly all corrections
   are for premature stops, not overshooting. OpenVLA tends to under-act rather
   than over-act, and the monitor nudges it forward.

3. **Single-axis corrections are critical.** The `go_above_pergola` task shows
   that OpenVLA executes forward motion well but misses the vertical component.
   The monitor's "ascend 0.5 meters" correction addresses the specific axis
   the drone is failing on, which would not be possible with a simple
   "try again" approach.

4. **Visual stopping conditions require the monitor.** Tasks like
   `turn_right_until_red_car` and `approach_person_ahead` include conditions
   ("until you see X", "until close to X") that OpenVLA cannot evaluate.
   Without the monitor, baselines stop immediately (~20 steps for turn_right).
   With the monitor, the system persists through multiple correction cycles
   until the VLM confirms the visual condition is met.

5. **Instruction conversion is important but minimal.** The `SubgoalConverter`
   strips visual conditions (e.g., "Turn right until you see the red car" →
   "turn right") so OpenVLA receives an actionable command. For tasks without
   conditions (e.g., "Go above the pergola"), the instruction passes through
   unchanged.

6. **Turn/rotation tasks are hardest.** `turn_right_until_red_car` uses the
   most corrections (9) and steps (199), with the drone needing many small
   rotation nudges. This may reflect OpenVLA's training distribution having
   fewer pure rotation demonstrations.

7. **Baseline behavior is inconsistent.** Several baselines hit `max_steps`
   (200) without converging, while others converge very quickly. Without a
   monitor, there is no principled termination criterion beyond pose stall.

### 8.3 Missing data

- `turn_right_until_red_car/llm_run_02` is absent, breaking the 3-replicate
  design for that task.
- Baseline runs have no `last_completion_pct` (no VLM evaluation), so
  quantitative success comparison with LLM runs requires post-hoc evaluation.
