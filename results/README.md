# Results

This directory contains the run records that back the numbers and qualitative
claims in the paper.

## What is committed

For every condition and the hardware run, the structured records:

- `run_info.json` -- the full task config, model versions, LLM decomposition
  output, every VLM call's tokens and timing, and the final stop reason.
- `subgoal_*/diary_summary.json` -- per-sub-goal: the full natural-language
  diary, the correction history, the override (peak-dropoff and force-converge)
  history, peak completion, total VLM calls.
- `trajectory_log.json` -- pose per step.
- `step_timings.jsonl` -- per-step wall-clock timings.
- `playback.mp4` (simulation runs only) -- low-res qualitative recording.

For the hardware run, the additional artifacts:

- `instruction.txt` -- the free-form mission instruction fed to the planner.
- `recording_log.jsonl` -- the wire-level command log.
- `replay_synced_compressed.mp4` -- diagnostic replay (1x speed, 720p, ~13 MB).
- `replay_synced_2x_compressed.mp4` -- same content at 2x speed (~7 MB).

For the C0 simulation run, sub-goal 3 ("go to the black car") additionally
includes its full `diary_artifacts/` directory (per-checkpoint and
per-convergence prompts, responses, image grids). This is the walkthrough that
Appendix A of the paper quotes verbatim, kept in the repo so any reader can
inspect the actual VLM inputs and outputs that produced the corrective
imperatives.

## What is excluded

Held out via `.gitignore` to keep the repo navigable:

- `**/frames/` -- raw per-step camera frames. Several gigabytes in aggregate.
  Re-running any condition regenerates them.
- `**/_replay_tmp/` -- intermediate working files for replay video synthesis.
- All `diary_artifacts/checkpoint_*` and `diary_artifacts/convergence_*` for
  runs other than C0 sub-goal 3. The aggregate text and outcomes for these
  checkpoints are already preserved in `diary_summary.json`.
- The original 1080p hardware replay videos (`replay_synced.mp4`,
  `replay_synced_2x.mp4`), the in-system `playback.mp4` (drone-cam only), and
  the raw 1080p phone recording. The compressed 720p replays preserve the
  qualitative content at a fraction of the size.

## Mapping paper claims to files

- Table 3 (per-condition results): each row corresponds to one
  `condition{N}/.../run_info.json`. The `subgoal_count`, `subgoal_summaries`,
  `aborted`, and `stop_reason` fields are authoritative.
- Table 4 (per-sub-goal C0 results): `condition0/.../subgoal_*/diary_summary.json`,
  fields `total_steps`, `corrections_used`, `vlm_call_count`,
  `peak_completion`, `last_completion_pct`, `stop_reason`.
- Section 5.4 hardware diary excerpts: `hardware/subgoal_*/diary_summary.json`,
  `diary` field.
- Appendix A walkthrough: `condition0/.../subgoal_03_go_to_the_black_car/`,
  in particular `diary_artifacts/checkpoint_0090/` (Appendix A.2 and A.3) and
  `diary_artifacts/convergence_002/` (Appendix A.4).
