# Experiment Plan: Ablation Studies for Neuro-Symbolic UAV Navigation

**Goal:** Produce the experiments and ablations needed to publish this system at a top-tier robotics conference (CoRL 2026 / RA-L / ICRA 2027). Each experiment isolates one component's contribution. The current evidence is n=1 on a single 6-subgoal mission; this plan fills that gap.

**What to build:** A script `scripts/run_experiments.py` that automates all experiment conditions, plus new task JSON files for mission diversity. The script should reuse the existing control-loop code from `scripts/run_integration.py`, `scripts/run_goal_adherence.py`, and `scripts/run_ltl.py` rather than duplicating it.

---

## Prerequisites

Before running any experiments, the executing session should confirm the following with the user:

1. **OpenVLA server is running** on port 5007 (`python scripts/start_server.py`)
2. **UnrealZoo Downtown West is running** (the sim binary)
3. **API keys are configured** in `.env.local` (OpenAI for gpt-5.4 and gpt-4o-mini)
4. **Approximate budget.** Each full-system run of the 6-subgoal mission takes ~45 min and makes ~280 VLM calls. The full plan is roughly 56-80 runs. Ask the user: "How many trials per condition? I recommend 5 for the long-horizon ablations and 3 for single-subgoal. 5+3 gives ~56 runs (~33 hours sim, ~16K VLM calls). Want to adjust?"
5. **Seed strategy.** Ask: "Should each trial use a different random seed (for NPC variation), or fixed seed across conditions (for controlled comparison)? I recommend fixed seed per trial index (seed=1 for trial 1 of every condition, seed=2 for trial 2, etc.) so each condition sees the same NPC layout."

---

## Experiment Structure

All results go under `results/experiments/` with the structure:

```
results/experiments/
  EXP1_full_system_replication/
    trial_01_seed1/
      run_info.json
      trajectory_log.json
      frames/
      subgoal_01_.../diary_summary.json
      ...
    trial_02_seed2/
    ...
  EXP2_ablation_no_monitor/
    trial_01_seed1/
    ...
  EXP3_ablation_no_planner/
    ...
  EXP4_ablation_raw_openvla/
    ...
  EXP5_vlm_model_sweep/
    gpt-5.4/trial_01_seed1/
    gpt-4o/trial_01_seed1/
    gpt-4o-mini/trial_01_seed1/
    ...
  EXP6_single_subgoal_isolation/
    approach_person_ahead/
      with_monitor/trial_01/
      baseline/trial_01/
    ...
  EXP7_mission_diversity/
    mission_02_3subgoal/trial_01_seed1/
    mission_03_4subgoal/trial_01_seed1/
    ...
  summary.json          # aggregate metrics across all experiments
```

---

## Experiment 1: Full System Replication (CRITICAL)

**Purpose:** Move from n=1 to n=5+ on the existing 6-subgoal mission with production config.

**Why it matters:** Every reviewer's first question. Without this, nothing else is credible.

**Condition:** Full system (LTL-NL planner + SubgoalConverter + LiveDiaryMonitor + corrective re-prompting).

**Task:** `tasks/system/first_task.json` (the existing 6-subgoal Downtown West mission).

**Config:**
- `llm_model=gpt-4o-mini`
- `monitor_model=gpt-5.4`
- `max_steps_per_subgoal=300`
- `diary_check_interval=10`
- `max_corrections=15`

**Trials:** 5 (or whatever the user confirms in prerequisites).

**Metrics to collect per trial:**
- Per-subgoal: outcome (complete/incomplete), steps, corrections_used, vlm_calls, peak_completion, stop_reason
- Per-trial: subgoals_completed / subgoals_attempted, total_steps, total_vlm_calls, total_corrections, wall_clock_seconds
- Per-subgoal: vlm_rtts list (from diary_summary.json) for latency analysis

**Metrics to aggregate:**
- Mean and std of subgoal completion rate across trials
- Per-subgoal-type completion rate (pass-by, turn-until, reach, fine-turn, landing)
- Mean and std of total steps, vlm_calls, corrections
- Mean and std of peak_completion on failed subgoals
- VLM latency: mean, median, p95 of per-call RTT (from vlm_rtts)

**Implementation notes:**
- This is essentially `run_integration.py --task first_task.json` run N times with different seeds.
- The script should loop over trial indices, set the seed, run the integration loop, and save to the appropriate directory.
- After all trials, compute and save aggregate metrics to `EXP1_full_system_replication/aggregate.json`.

---

## Experiment 2: Ablation -- No Diary Monitor (CRITICAL)

**Purpose:** Isolate the LiveDiaryMonitor's contribution. This is the "single most important missing experiment" per the long draft.

**Condition:** LTL-NL planner + SubgoalConverter, but NO diary monitor. Subgoals advance on OpenVLA's native convergence detector (the pose-stall heuristic: 3cm/1deg/10 consecutive steps). No VLM checkpoints, no corrective commands.

**Task:** Same `first_task.json`.

**Config:**
- Same as EXP1 except: diary monitor disabled.
- Subgoal transitions happen only on pose-stall convergence or max_steps.
- The SubgoalConverter still converts subgoals (so OpenVLA gets the same clean imperatives).

**Implementation notes:**
- This is closest to what `run_ltl.py` already does: LTL decomposition, sequential subgoal execution, convergence-only transitions.
- The script should run the LTL planner to decompose the instruction, then for each subgoal: run SubgoalConverter, send the instruction to OpenVLA, run the control loop with ONLY convergence detection (no `LiveDiaryMonitor.on_frame()`, no `on_convergence()` supervisor calls).
- On convergence or max_steps, advance to the next subgoal unconditionally.
- Save the same per-subgoal metrics (minus VLM-specific ones: vlm_calls=0, corrections=0, peak_completion=N/A).

**Trials:** Same count as EXP1 (5), same seeds.

**Expected result (from long draft):** 30+ percentage-point drop in subgoal success rate vs. EXP1, principally on pass-by-landmark subgoals where the diary monitor catches premature convergence.

---

## Experiment 3: Ablation -- No LTL Planner

**Purpose:** Isolate the LTL-NL decomposition's contribution.

**Condition:** Send the entire raw mission instruction to OpenVLA as a single instruction, supervised by a single LiveDiaryMonitor that treats the whole mission as one subgoal.

**Task:** Same `first_task.json` instruction, but no LTL decomposition.

**Config:**
- No LTL planner, no SubgoalConverter.
- `monitor_model=gpt-5.4`
- The diary monitor receives the full raw instruction as its subgoal_text.
- `max_steps=1800` (sum of 6 x 300, so the budget matches the full system's total budget).
- `max_corrections=90` (sum of 6 x 15).
- `diary_check_interval=10`

**Implementation notes:**
- Skip the LTL planning step entirely.
- Create a single LiveDiaryMonitor with the raw instruction.
- Run the standard control loop (with diary checkpoints and convergence supervision) for up to max_steps.
- The monitor will attempt to track completion of a compound 6-part instruction as a single subgoal.

**Trials:** Same count as EXP1 (5), same seeds.

**Expected result:** Catastrophic failure beyond the first one or two actions. The diary monitor is not designed to track multi-stage progress in a single subgoal and will likely hover around 0.4-0.6 completion indefinitely.

---

## Experiment 4: Ablation -- Raw OpenVLA (Baseline)

**Purpose:** Show what happens with no system components at all.

**Condition:** Send the entire raw mission instruction directly to OpenVLA. No LTL planner, no SubgoalConverter, no diary monitor, no convergence supervision. Just the raw policy.

**Task:** Same `first_task.json` instruction.

**Config:**
- No LTL planner, no SubgoalConverter, no LiveDiaryMonitor.
- Raw instruction sent to OpenVLA as-is.
- `max_steps=1800` (same total budget).
- Convergence detection only (pose-stall, same thresholds as EXP2).

**Implementation notes:**
- Simplest possible control loop: capture frame, send to OpenVLA with the raw instruction, apply action, check for convergence.
- No VLM calls at all.
- Log trajectory and frames for post-hoc analysis.
- Since there are no subgoals, "success" is judged qualitatively (or by running an offline VLM evaluation over the trajectory after the fact). To make this quantitative, after the run completes, use the offline goal-adherence checker (`check_subtask_completed_diary` from `rvln.ai.utils.goal_adherence`) to score how many of the 6 subgoals were achieved based on the saved frames. This post-hoc scoring uses the same VLM but does not influence the control loop.

**Trials:** Same count as EXP1 (5), same seeds.

**Expected result:** Completion of the first 1-2 actions (the drone will move forward and maybe turn), then drift. The policy has no mechanism to sequence 6 ordered subgoals from a compound instruction.

---

## Experiment 5: VLM Sensitivity Sweep

**Purpose:** Characterize the success-rate-vs-cost frontier across VLM providers. Answers "what if I cannot afford gpt-5.4?"

**Condition:** Full system (same as EXP1), but vary the monitor_model.

**Models to test:**
- `gpt-5.4` (production, already covered by EXP1; reuse those results)
- `gpt-4o`
- `gpt-4o-mini`

Ask the user: "Do you want to include non-OpenAI models (gemini-2.5-flash, claude-sonnet-4-6)? Each adds 5 more runs. I recommend starting with the three OpenAI tiers and adding others only if needed."

**Task:** Same `first_task.json`.

**Config:** Same as EXP1 except `monitor_model` varies.

**Trials:** 3 per model (cheaper models are less consistent, but 3 trials is enough to see the trend). Reuse EXP1's first 3 trials for the gpt-5.4 condition.

**Metrics:** Same as EXP1, plus per-call VLM cost estimate (based on token counts if available, or flat per-call pricing).

**Implementation notes:**
- For each model, run the full integration loop 3 times.
- If a model is not available (e.g., API key missing), skip it and log a warning.

---

## Experiment 6: Single-Subgoal Isolation Study

**Purpose:** Isolate the diary monitor's contribution at the individual subgoal level, complementing the long-horizon ablation (EXP1 vs EXP2).

**Condition:** Two conditions per task:
1. **Baseline (no monitor):** Raw subgoal string sent directly to OpenVLA. No SubgoalConverter, no diary monitor. Convergence-only termination.
2. **With monitor:** SubgoalConverter + LiveDiaryMonitor with corrections.

**Tasks:** The 6 existing tasks in `tasks/goal_adherence/`:
- `approach_person_ahead.json`
- `approach_white_building.json`
- `go_above_pergola.json`
- `go_between_tree_and_streetlight.json`
- `move_through_pergola.json`
- `turn_right_until_red_car.json`

**Config:** As specified in each task JSON. `monitor_model=gpt-5.4`.

**Trials:** 3 per condition per task (6 tasks x 2 conditions x 3 trials = 36 runs).

**Metrics:**
- Binary success (did the subgoal complete?)
- Steps to completion (or max_steps if incomplete)
- Peak completion percentage (monitor condition only)
- Corrections used (monitor condition only)
- VLM calls (monitor condition only)

**Implementation notes:**
- This is what `run_goal_adherence.py` already does. The script should wrap it with proper trial/seed management and save results in the experiment directory structure.

---

## Experiment 7: Mission Diversity

**Purpose:** Show the system generalizes beyond a single mission. Without this, external validity is limited.

**New tasks to author:** Create 4 additional multi-step missions in `tasks/system/` that vary:
- Number of subgoals (3, 4, 5, 8)
- Subgoal type mix (some heavy on reach, some heavy on turn-until-detect, some mixed)
- Different starting positions in Downtown West

The executing session should author these task JSONs. Here are the specifications:

### Mission 2: Short mission (3 subgoals)
```json
{
  "instruction": "Turn left until you see the white building, then fly toward the white building, then land.",
  "initial_pos": [-86, 8532, 559, -65],
  "max_steps_per_subgoal": 300,
  "diary_check_interval": 10,
  "max_corrections": 15
}
```
**Subgoal types:** turn-until-detect, reach, landing.

### Mission 3: Medium mission (4 subgoals)
```json
{
  "instruction": "Go forward past the tree, then turn right until you see the person, then approach the person, then go above the pergola.",
  "initial_pos": [-9435, 15, 296, 7],
  "max_steps_per_subgoal": 300,
  "diary_check_interval": 10,
  "max_corrections": 15
}
```
**Subgoal types:** pass-by, turn-until-detect, reach, altitude-change.

### Mission 4: Medium mission with spatial precision (5 subgoals)
```json
{
  "instruction": "Go between the tree and the streetlight, then move forward past the streetlight, then turn left until you see the building, then go to the building, then descend and land.",
  "initial_pos": [-9435, 15, 296, 7],
  "max_steps_per_subgoal": 300,
  "diary_check_interval": 10,
  "max_corrections": 15
}
```
**Subgoal types:** between, pass-by, turn-until-detect, reach, landing.

### Mission 5: Long mission (8 subgoals)
Ask the user: "I need to author an 8-subgoal mission. Can you help me pick a starting position and describe a route through Downtown West that touches 8 distinct landmarks? Or should I construct one from combinations of the existing goal-adherence task locations?"

**Condition:** Full system (same as EXP1).

**Trials:** 3 per mission.

**Metrics:** Same as EXP1. Additionally:
- Per-mission completion rate (subgoals completed / subgoals attempted)
- Per-subgoal-type completion rate aggregated across all missions
- Correlation between mission length and completion rate

---

## Implementation: `scripts/run_experiments.py`

The script should:

1. **Accept a `--experiment` flag** to select which experiment to run:
   - `--experiment EXP1` through `--experiment EXP7`, or `--experiment all`
   - `--experiment EXP1,EXP2` for comma-separated subsets

2. **Accept `--trials N`** to override the default trial count.

3. **Accept `--seeds 1,2,3,4,5`** to specify seed list (default: 1 through N).

4. **Accept `--dry-run`** to print what would be run without running it (trial count, estimated time, estimated VLM calls).

5. **Reuse existing code** rather than duplicating control loops:
   - EXP1, EXP5, EXP7: call `run_integrated_control_loop()` from `run_integration.py`
   - EXP2: adapt the control loop from `run_ltl.py` (LTL decomposition + convergence-only)
   - EXP3: single-subgoal variant of `run_integration.py` (one LiveDiaryMonitor, no LTL)
   - EXP4: minimal control loop (OpenVLA only, no monitor)
   - EXP6: call the experiment runner from `run_goal_adherence.py`

6. **Skip completed trials.** If `results/experiments/EXP1_full_system_replication/trial_01_seed1/run_info.json` already exists, skip that trial. This allows resuming after interruption.

7. **Compute aggregates** after each experiment completes. Save to `aggregate.json` in each experiment directory with:
   - Per-condition means and standard deviations for all metrics
   - Per-subgoal-type breakdown
   - VLM latency statistics (from vlm_rtts in diary summaries)

8. **Generate a summary table** after all experiments. Save to `results/experiments/summary.json` and print a human-readable table to stdout:

```
Experiment                    | Trials | Subgoal Success | Steps (mean) | VLM Calls | Corrections
EXP1 Full System              |   5    |   52% +/- 8%    |  1180 +/- 90 |  275 +/- 20 |  34 +/- 5
EXP2 No Monitor               |   5    |   22% +/- 10%   |  1400 +/- 50 |     0       |     0
EXP3 No Planner               |   5    |    8% +/- 5%    |  1800 +/- 0  |   45 +/- 8  |   6 +/- 3
EXP4 Raw OpenVLA              |   5    |    5% +/- 5%    |  1800 +/- 0  |     0       |     0
```

---

## Run Order (Priority)

The experiments are ordered by reviewer importance. If time or budget is limited, stop after whichever step you can afford:

1. **EXP1** (Full system replication) -- ~4 hours. Non-negotiable.
2. **EXP2** (No monitor ablation) -- ~4 hours. Non-negotiable.
3. **EXP4** (Raw OpenVLA baseline) -- ~4 hours. Needed for the ablation table.
4. **EXP3** (No planner ablation) -- ~4 hours. Completes the 4-condition ablation table.
5. **EXP6** (Single-subgoal isolation) -- ~18 hours. Strong supporting evidence.
6. **EXP5** (VLM sweep) -- ~4.5 hours. Answers cost question.
7. **EXP7** (Mission diversity) -- ~9 hours. Generalization evidence.

**Minimum viable set for submission:** EXP1 + EXP2 + EXP4 (~12 hours).
**Recommended set:** EXP1 through EXP6 (~34 hours).
**Full set:** All experiments (~47 hours).

---

## Post-Experiment Analysis

After experiments complete, the executing session should:

1. **Update `long_draft.md` Section 7.2** with the new multi-trial numbers from EXP1.
2. **Add a new Section 7.3** (or update the existing one) with the ablation comparison table (EXP1 vs EXP2 vs EXP3 vs EXP4).
3. **Update Section 9.1** to remove the "single run" limitation.
4. **Generate completion-timeline plots** (TODO-8 from the long draft): per-subgoal completion percentage vs. checkpoint, with correction events marked. Use matplotlib, save to `results/experiments/figures/`.
5. **Report VLM latency** (TODO-16): extract RTT statistics from EXP1's vlm_rtts and add a Section 7.7 "Inference Latency and Cost" to the long draft.

---

## Clarifying Questions for the Executing Session

Before starting implementation, the executing session should ask:

1. "How many trials per condition? (Recommended: 5 for EXP1-4, 3 for EXP5-7)"
2. "Fixed seed per trial index across conditions, or random seeds? (Recommended: fixed)"
3. "Which experiments to run? (Recommended: EXP1 through EXP6 for ~34 hours)"
4. "Include non-OpenAI models in EXP5? (Recommended: start with OpenAI-only)"
5. "For EXP7 Mission 5 (8 subgoals): can you describe an 8-landmark route, or should I construct one from existing locations?"
6. "Is the OpenVLA server and UnrealZoo sim currently running?"

---

## File Inventory

Files to create:
- `scripts/run_experiments.py` -- main experiment runner
- `tasks/system/second_task.json` -- Mission 2 (3 subgoals)
- `tasks/system/third_task.json` -- Mission 3 (4 subgoals)
- `tasks/system/fourth_task.json` -- Mission 4 (5 subgoals)
- `tasks/system/fifth_task.json` -- Mission 5 (8 subgoals, needs user input)

Files to modify:
- None. The script imports from existing modules.

Files to read (for implementation reference):
- `scripts/run_integration.py` -- full system control loop to reuse
- `scripts/run_goal_adherence.py` -- single-subgoal control loop to reuse
- `scripts/run_ltl.py` -- LTL-only control loop to reuse for EXP2
- `src/rvln/ai/diary_monitor.py` -- LiveDiaryMonitor API
- `src/rvln/ai/ltl_planner.py` -- LTLSymbolicPlanner API
- `src/rvln/ai/subgoal_converter.py` -- SubgoalConverter API
- `src/rvln/config.py` -- default hyperparameters
- `src/rvln/sim/env_setup.py` -- simulation setup utilities
