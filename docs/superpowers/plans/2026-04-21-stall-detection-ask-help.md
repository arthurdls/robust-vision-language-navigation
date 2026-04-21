# Stall Detection: Ask-for-Help Mechanism

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When checkpoint completion percentage plateaus over several consecutive checkpoints, pause the drone and ask the human operator for a new instruction.

**Architecture:** Track completion percentage history in `LiveDiaryMonitor`. After each checkpoint, check if the last N values are within a small delta (stall). If stalled, return a new `"ask_help"` action. The `run_subgoal` loop in `interface.py` handles this by pausing the drone, printing status, and prompting for human input via stdin.

**Tech Stack:** Python, existing `LiveDiaryMonitor` / `DiaryCheckResult` / `run_subgoal` infrastructure.

---

### Task 1: Add stall detection to `LiveDiaryMonitor`

**Files:**
- Modify: `src/rvln/ai/diary_monitor.py:48-54` (DiaryCheckResult docstring)
- Modify: `src/rvln/ai/diary_monitor.py:226-257` (`__init__`)
- Modify: `src/rvln/ai/diary_monitor.py:587-666` (`_run_checkpoint`)
- Modify: `src/rvln/ai/diary_monitor.py:668-766` (`_run_checkpoint_async`)
- Create: `tests/test_stall_detection.py`

- [ ] **Step 1: Write failing test for stall detection helper**

```python
# tests/test_stall_detection.py
from rvln.ai.diary_monitor import LiveDiaryMonitor


def test_no_stall_when_not_enough_history():
    """Stall detection needs at least stall_window checkpoints."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.1, 0.12]
    m._stall_window = 3
    m._stall_threshold = 0.05
    assert m._is_stalled() is False


def test_stall_detected_when_flat():
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.30, 0.31, 0.32]
    m._stall_window = 3
    m._stall_threshold = 0.05
    assert m._is_stalled() is True


def test_no_stall_when_progressing():
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.30, 0.40, 0.50]
    m._stall_window = 3
    m._stall_threshold = 0.05
    assert m._is_stalled() is False


def test_no_stall_when_completion_high():
    """Don't ask for help if already nearly done."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.85, 0.86, 0.86]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is False


def test_stall_only_looks_at_last_window():
    """Earlier progress doesn't mask a recent plateau."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.10, 0.20, 0.30, 0.31, 0.32]
    m._stall_window = 3
    m._stall_threshold = 0.05
    assert m._is_stalled() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && python -m pytest tests/test_stall_detection.py -v`
Expected: FAIL because `_is_stalled` does not exist yet.

- [ ] **Step 3: Add `_is_stalled()` method and new init parameters**

In `src/rvln/ai/diary_monitor.py`, update the `DiaryCheckResult` docstring (line 49):

```python
@dataclass
class DiaryCheckResult:
    action: str           # "continue", "stop", "override", "command", "ask_help", or "force_converge"
    new_instruction: str  # populated when action is "override" or "command"
    reasoning: str
    diary_entry: str      # latest diary entry (what changed)
    completion_pct: float = 0.0  # latest estimated completion percentage
```

In `__init__` (after line 256, `self._last_displacement`), add the stall detection state and accept new parameters:

```python
# In __init__ signature, add these parameters after check_interval_s:
#   stall_window: int = 3,
#   stall_threshold: float = 0.05,
#   stall_completion_floor: float = 0.8,

# In __init__ body, after self._last_displacement line:
self._stall_window = stall_window
self._stall_threshold = stall_threshold
self._stall_completion_floor = stall_completion_floor
self._completion_history: List[float] = []
```

Add the `_is_stalled()` method (after `_format_displacement`, around line 572):

```python
def _is_stalled(self) -> bool:
    """Return True if completion has plateaued over the last stall_window checkpoints."""
    history = self._completion_history
    if len(history) < self._stall_window:
        return False
    recent = history[-self._stall_window:]
    if min(recent) >= self._stall_completion_floor:
        return False
    return max(recent) - min(recent) < self._stall_threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && python -m pytest tests/test_stall_detection.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_stall_detection.py src/rvln/ai/diary_monitor.py
git commit -m "Add _is_stalled() heuristic to LiveDiaryMonitor"
```

---

### Task 2: Wire stall detection into checkpoint results

**Files:**
- Modify: `src/rvln/ai/diary_monitor.py:587-666` (`_run_checkpoint`)
- Modify: `src/rvln/ai/diary_monitor.py:668-766` (`_run_checkpoint_async`)

- [ ] **Step 1: Write failing test for checkpoint returning ask_help**

Add to `tests/test_stall_detection.py`:

```python
from unittest.mock import patch, MagicMock
from pathlib import Path


def _make_monitor_with_history(history, stall_window=3, stall_threshold=0.05):
    """Build a LiveDiaryMonitor with pre-loaded completion history for testing."""
    m = LiveDiaryMonitor(
        subgoal="move forward",
        check_interval=2,
        model="gpt-4o",
        stall_window=stall_window,
        stall_threshold=stall_threshold,
    )
    m._completion_history = list(history)
    return m


@patch("rvln.ai.diary_monitor.query_vlm")
@patch("rvln.ai.diary_monitor.build_frame_grid")
@patch("rvln.ai.diary_monitor.sample_frames_every_n")
def test_checkpoint_returns_ask_help_on_stall(mock_sample, mock_grid, mock_vlm):
    """When completion has plateaued, _run_checkpoint should return ask_help."""
    m = _make_monitor_with_history([0.30, 0.31])

    # Set up frame paths so the checkpoint can run
    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    # Local query returns change text, global returns flat completion
    mock_vlm.side_effect = [
        "No visible change.",
        '{"complete": false, "completion_percentage": 0.32, "on_track": true, "should_stop": false}',
    ]

    result = m._run_checkpoint()
    assert result.action == "ask_help"
    assert "stall" in result.reasoning.lower() or "stall" in result.reasoning.lower()


@patch("rvln.ai.diary_monitor.query_vlm")
@patch("rvln.ai.diary_monitor.build_frame_grid")
@patch("rvln.ai.diary_monitor.sample_frames_every_n")
def test_checkpoint_returns_continue_when_not_stalled(mock_sample, mock_grid, mock_vlm):
    """Normal progress should still return continue."""
    m = _make_monitor_with_history([0.10, 0.20])

    m._frame_paths = [Path(f"/tmp/f{i}.png") for i in range(4)]
    m._frame_timestamps = [float(i) for i in range(4)]
    m._step = 4

    mock_grid.return_value = MagicMock()
    mock_sample.return_value = m._frame_paths[-4:]
    mock_vlm.side_effect = [
        "Moved closer to target.",
        '{"complete": false, "completion_percentage": 0.35, "on_track": true, "should_stop": false}',
    ]

    result = m._run_checkpoint()
    assert result.action == "continue"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && python -m pytest tests/test_stall_detection.py::test_checkpoint_returns_ask_help_on_stall -v`
Expected: FAIL because `_run_checkpoint` does not yet use `_is_stalled()`.

- [ ] **Step 3: Add stall check after global response in `_run_checkpoint`**

In `_run_checkpoint` (line 656-665), after `result = self._parse_global_response(...)`, add the stall check before returning. Replace the block from line 656 to 666:

```python
        result = self._parse_global_response(
            response_global, diary_entry, parsed=parsed,
        )

        self._last_completion_pct = result.completion_pct
        self._peak_completion = max(self._peak_completion, result.completion_pct)
        self._completion_history.append(result.completion_pct)
        self._diary.append(
            f"Checkpoint {step}: completion = {result.completion_pct:.2f}"
        )

        if result.action == "continue" and self._is_stalled():
            return DiaryCheckResult(
                action="ask_help",
                new_instruction="",
                reasoning=f"Stall detected: completion plateau over last {self._stall_window} checkpoints.",
                diary_entry=diary_entry,
                completion_pct=result.completion_pct,
            )

        return result
```

Apply the same pattern in `_run_checkpoint_async` (line 756-765), replacing the equivalent block:

```python
        result = self._parse_global_response(
            response_global, diary_entry, parsed=parsed,
        )

        self._last_completion_pct = result.completion_pct
        self._peak_completion = max(self._peak_completion, result.completion_pct)
        self._completion_history.append(result.completion_pct)
        self._diary.append(
            f"Checkpoint ~{step}: completion = {result.completion_pct:.2f}"
        )

        if result.action == "continue" and self._is_stalled():
            return DiaryCheckResult(
                action="ask_help",
                new_instruction="",
                reasoning=f"Stall detected: completion plateau over last {self._stall_window} checkpoints.",
                diary_entry=diary_entry,
                completion_pct=result.completion_pct,
            )

        return result
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && python -m pytest tests/test_stall_detection.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rvln/ai/diary_monitor.py tests/test_stall_detection.py
git commit -m "Wire stall detection into checkpoint results as ask_help action"
```

---

### Task 3: Handle `ask_help` in `run_subgoal`

**Files:**
- Modify: `src/rvln/mininav/interface.py:603-894` (`run_subgoal` function)
- Modify: `src/rvln/mininav/interface.py:897-973` (`parse_args`)

- [ ] **Step 1: Add `stall_*` parameters to `run_subgoal` signature and pass to `LiveDiaryMonitor`**

In `run_subgoal` (line 603), add three new parameters after `check_interval_s`:

```python
def run_subgoal(
    subgoal_nl: str,
    subgoal_index: int,
    run_dir: Path,
    frames_dir: Path,
    openvla: OpenVLAClient,
    camera: ThreadedCamera,
    control: DroneControlClient,
    pose_manager: PoseManager,
    monitor_model: str,
    check_interval: int,
    max_steps: int,
    max_corrections: int,
    frame_offset: int,
    command_dt_s: float,
    action_pose_mode: str,
    trajectory_log: List[Dict[str, Any]],
    check_interval_s: Optional[float] = None,
    stall_window: int = 3,
    stall_threshold: float = 0.05,
    stall_completion_floor: float = 0.8,
) -> Dict[str, Any]:
```

Pass them through to `LiveDiaryMonitor` (line 632-639):

```python
    monitor = LiveDiaryMonitor(
        subgoal=subgoal_nl,
        check_interval=check_interval,
        model=monitor_model,
        artifacts_dir=diary_artifacts,
        max_corrections=max_corrections,
        check_interval_s=check_interval_s,
        stall_window=stall_window,
        stall_threshold=stall_threshold,
        stall_completion_floor=stall_completion_floor,
    )
```

- [ ] **Step 2: Add `ask_help` handler in async mode (poll result block)**

In the async poll block (after the `force_converge` handler, around line 705), add handling for `ask_help`:

```python
                if async_result.action == "ask_help":
                    control.send_command(frame_offset + step, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
                    logger.warning(
                        "Stall detected at step %d (completion: %.0f%%). Asking operator for help.",
                        step, async_result.completion_pct * 100,
                    )
                    print(f"\n{'='*60}")
                    print(f"STALL DETECTED - subgoal: {subgoal_nl}")
                    print(f"Completion: {async_result.completion_pct:.0%}")
                    print(f"Current instruction: {current_instruction}")
                    print(f"Reasoning: {async_result.reasoning}")
                    print(f"{'='*60}")
                    human_input = input("New instruction (or 'skip' to continue, 'abort' to stop): ").strip()
                    if human_input.lower() == "abort":
                        stop_reason = "operator_abort"
                        total_steps = step
                        break
                    if human_input and human_input.lower() != "skip":
                        override_history.append({
                            "step": step,
                            "type": "operator_help",
                            "old_instruction": current_instruction,
                            "new_instruction": human_input,
                            "reasoning": async_result.reasoning,
                        })
                        current_instruction = human_input
                        openvla_pose_origin = list(subgoal_rel_pose)
                        small_count = 0
                        last_pose = None
                        last_correction_time = time.time()
                        last_correction_step = step
                        openvla.reset_model()
```

- [ ] **Step 3: Add `ask_help` handler in sync mode (on_frame result block)**

In the sync mode block (after `force_converge` handling around line 735), add:

```python
            if result.action == "ask_help":
                control.send_command(frame_offset + step, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
                logger.warning(
                    "Stall detected at step %d (completion: %.0f%%). Asking operator for help.",
                    step, result.completion_pct * 100,
                )
                print(f"\n{'='*60}")
                print(f"STALL DETECTED - subgoal: {subgoal_nl}")
                print(f"Completion: {result.completion_pct:.0%}")
                print(f"Current instruction: {current_instruction}")
                print(f"Reasoning: {result.reasoning}")
                print(f"{'='*60}")
                human_input = input("New instruction (or 'skip' to continue, 'abort' to stop): ").strip()
                if human_input.lower() == "abort":
                    stop_reason = "operator_abort"
                    total_steps = step
                    break
                if human_input and human_input.lower() != "skip":
                    override_history.append({
                        "step": step,
                        "type": "operator_help",
                        "old_instruction": current_instruction,
                        "new_instruction": human_input,
                        "reasoning": result.reasoning,
                    })
                    current_instruction = human_input
                    openvla_pose_origin = list(subgoal_rel_pose)
                    small_count = 0
                    last_pose = None
                    last_correction_step = step
                    openvla.reset_model()
```

- [ ] **Step 4: Add CLI args for stall detection and wire them through `main()`**

In `parse_args` (after the `--max_corrections` arg, around line 939), add:

```python
    parser.add_argument("--stall_window", type=int, default=3,
        help="Number of consecutive checkpoints with flat completion to trigger help request.")
    parser.add_argument("--stall_threshold", type=float, default=0.05,
        help="Max completion delta across stall_window checkpoints to count as stalled.")
    parser.add_argument("--stall_completion_floor", type=float, default=0.8,
        help="Don't trigger stall detection above this completion level.")
```

In `main()`, pass the new args to `run_subgoal` (around line 1083-1101):

```python
            result = run_subgoal(
                subgoal_nl=current_subgoal,
                subgoal_index=subgoal_index,
                run_dir=run_dir,
                frames_dir=frames_dir,
                openvla=openvla,
                camera=camera,
                control=control,
                pose_manager=pose_manager,
                monitor_model=monitor_model,
                check_interval=args.diary_check_interval,
                max_steps=args.max_steps_per_subgoal,
                max_corrections=args.max_corrections,
                frame_offset=frame_offset,
                command_dt_s=args.command_dt_s,
                action_pose_mode=args.action_pose_mode,
                trajectory_log=trajectory_log,
                check_interval_s=check_interval_s,
                stall_window=args.stall_window,
                stall_threshold=args.stall_threshold,
                stall_completion_floor=args.stall_completion_floor,
            )
```

- [ ] **Step 5: Run all tests**

Run: `cd /home/arthurdls/SuperUROP/rvln-adls && python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rvln/ai/diary_monitor.py src/rvln/mininav/interface.py tests/test_stall_detection.py
git commit -m "Handle ask_help action in run_subgoal with operator prompt and CLI args"
```
