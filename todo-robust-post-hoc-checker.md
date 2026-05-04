# Post-Hoc Constraint Checker with gpt-5.4

## Problem

During experiments, constraint violations are detected at runtime by the GoalAdherenceMonitor
using VLM frame analysis. This is inherently noisy: the VLM may miss violations or hallucinate
them. We need a post-hoc verification pass to validate constraint adherence on completed
episodes, producing ground-truth labels for the paper's analysis.

## Proposed Solution

A standalone script that replays each episode's saved frames through gpt-5.4 and checks
whether constraints were actually upheld. This gives a second, independent signal that can
be compared against the runtime monitor's judgments.

## Design

### Inputs
- Episode result directory (contains saved frames, episode metadata JSON)
- Task JSON (contains instruction, expected constraints via LTL planner)

### Process
1. Load the task's LTL formula and classify constraints (using the formula-structural
   classifier in `ltl_planner.py`).
2. For each constraint, construct a yes/no visual question (e.g., "Is the drone above
   the clear trail?" for a positive constraint, "Is the drone on the road?" for a negative
   constraint).
3. Sample frames at regular intervals from the episode (e.g., every 10th frame).
4. Send each sampled frame + constraint question to gpt-5.4.
5. Record per-frame, per-constraint verdicts.
6. Aggregate: a constraint is violated if any frame shows a violation (positive constraint
   not maintained, or negative constraint triggered).

### Outputs
- Per-episode JSON with frame-level constraint verdicts
- Summary CSV: episode_id, constraint, violation_count, violation_frames, verdict
- Comparison with runtime monitor judgments (agreement rate)

## Steps

1. Write `scripts/post_hoc_constraint_check.py`.
2. Add a frame sampling utility (handle variable episode lengths).
3. Build the constraint-to-question mapper (reuse `ConstraintInfo.description` and
   `ConstraintInfo.polarity` from `ltl_planner.py`).
4. Implement the gpt-5.4 query loop with rate limiting and retry logic.
5. Add aggregation and comparison logic.
6. Test on a small set of completed episodes before running on the full dataset.

## Notes

- This is for analysis only, not runtime. It runs after all experiments complete.
- Budget: at ~10 frames per episode and 315 episodes, this is ~3150 VLM calls. With
  multiple constraints per constrained task, estimate ~5000-8000 total calls.
- Consider batching frames into a single multi-image prompt per episode to reduce calls.
