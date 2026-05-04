# TODO: Remaining Work for Full Submission

## Experimental Gaps (Priority Order)

### High Priority
- [ ] **Multi-run replication**: Run all 7 conditions multiple times (at least 3-5 runs per condition) on the current task to get error bars and statistical significance
- [ ] **Task diversity**: Design and run 8-12 additional long-horizon missions varying in number of subgoals (3-8), subgoal type mix, and scene region
- [ ] **Constraint tasks**: Design and run tasks with temporal constraints (avoidance and maintenance) to demonstrate the LTL formalism's advantage over LLM-only planning (Condition 2). This is the key differentiator. At least 6-9 tasks with constraints per the experimental design (60% of tasks should have constraints)
- [ ] **Starting position variations**: Run each task with 3 starting positions/orientations per the experimental design (default, rotated 90 degrees, offset 5-10m)

### Medium Priority
- [ ] **Hardware deployment**: Wire MiniNav interface to full pipeline and run integrated long-horizon mission on custom quadcopter
- [ ] **Post-hoc constraint checker**: Implement script to evaluate constraint adherence for Conditions 1 and 3 (no monitor) by passing trajectory frames through VLM
- [ ] **Wall-clock / latency analysis**: Collect and report per-call VLM RTT numbers from production runs (instrumentation exists in _timed_query_vlm)
- [ ] **VLM sensitivity sweep**: Re-run integrated mission with monitor_model in {gpt-5.4, gpt-4o, gpt-4o-mini, gemini-2.x} to characterize success-rate-vs-cost frontier

### Lower Priority
- [ ] **Goal-adherence single-subgoal ablation**: Re-run the 6 single-subgoal tasks under final monitor configuration with N=10 trials per condition
- [ ] **Diary length sweep**: Vary diary_check_interval in {5, 10, 20, 30} and MAX_GLOBAL_FRAMES in {6, 9, 12}
- [ ] **Correction vocabulary expansion**: Add compositional corrective imperatives and re-run spatial-precision failure cases
- [ ] **Convergence detector hyperparameter sweep**: Tune the 3cm / 1 degree / 10-cycle thresholds

## Paper Gaps
- [ ] **Render Figure 1**: Method-at-a-glance teaser diagram (input to output pipeline)
- [ ] **Create comparison figures**: Qualitative trajectory comparisons across conditions
- [ ] **Extract frame sequences**: Pull representative frames from results for paper figures
- [ ] **Verify prompts**: Re-verify all appendix prompts against src/rvln/ai/prompts.py at submission time

## Change to remember
- [ ] **Update to diagram**: Update the system diagram so that subgoal converter does not ask for help from user
