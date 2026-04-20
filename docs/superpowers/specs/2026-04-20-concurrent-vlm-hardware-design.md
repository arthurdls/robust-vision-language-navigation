# Concurrent VLM Monitor with Hardware Command Loop

## Problem

The current `run_subgoal` loop is fully sequential. Each step blocks on
`openvla.predict()` (~0.2s), sends commands, then blocks on
`monitor.on_frame()` (1-3s every N steps). During VLM calls the hardware
receives no commands, and the variable gap inflates `dt_s` on the mock
server, distorting pose integration.

## Goal

Keep sending OpenVLA commands to hardware at inference speed (~0.2s) while
VLM monitor checkpoints run concurrently in the background. Checkpoint
interval is time-based (default 1s) for hardware, frame-based for
simulation. Changes must not affect simulation code.

## Design

### LiveDiaryMonitor: time-based mode with internal thread

**New constructor parameter:**

```python
check_interval_s: Optional[float] = None
```

When `check_interval_s` is set, the monitor operates in time-based async
mode. When `None` (default), it operates in the existing frame-based
synchronous mode. Both cannot be active simultaneously.

**Internal state (protected by `threading.Lock`):**

- `_latest_frame_path: Optional[Path]` - updated by `on_frame` every step
- `_latest_displacement: Optional[List[float]]` - updated by `on_frame`
  every step
- `_pending_result: Optional[DiaryCheckResult]` - written by monitor
  thread, read/cleared by `poll_result()`
- `_convergence_request: Optional[Tuple[Path, List[float]]]` - written by
  `request_convergence()`, consumed by monitor thread
- `_last_checkpoint_time: float` - tracks when the last checkpoint ran

**Background thread (time-based mode only):**

Spawned in the constructor when `check_interval_s` is set. Runs a loop:

1. Sleep until `check_interval_s` has elapsed since `_last_checkpoint_time`
2. Check for convergence request first (higher priority):
   - If `_convergence_request` is set, grab it, run `_run_convergence()`
     (existing `on_convergence` logic), store result in `_pending_result`
3. Otherwise, grab `_latest_frame_path` and `_latest_displacement` under
   the lock
4. Run `_run_checkpoint()` (2 VLM calls, 1-3s total)
5. Store result in `_pending_result`, update `_last_checkpoint_time`

The thread exits when a `_stop_event` (`threading.Event`) is set.

**`on_frame` behavior in time-based mode:**

- Acquires lock, updates `_latest_frame_path` and `_latest_displacement`
- Appends frame path to `_frame_paths` (for grid sampling)
- Increments `_step`
- Returns immediately with `action="continue"` (never blocks on VLM)

**`on_frame` behavior in frame-based mode (unchanged):**

- Identical to current implementation
- Runs checkpoint synchronously every `check_interval` frames

**New public methods:**

```python
def poll_result(self) -> Optional[DiaryCheckResult]:
    """Return and clear any pending async result. Returns None if no
    result is ready or if operating in frame-based mode."""

def request_convergence(
    self, frame_path: Path, displacement: List[float]
) -> None:
    """Queue a convergence check (non-blocking). The monitor thread
    picks it up on its next cycle. Only valid in time-based mode."""
```

`poll_result()` in frame-based mode always returns `None`, so callers can
use it unconditionally without branching on mode.

**Frame sampling for the VLM grid (time-based mode):**

The monitor stores `(path, timestamp)` tuples alongside frame paths. When
building the grid for a checkpoint, it selects frames closest to each
`check_interval_s` boundary (e.g., at t=0s, t=1s, t=2s, ...) rather than
every Nth frame. This ensures the VLM sees frames spaced at consistent
real-time intervals regardless of how many OpenVLA steps occurred.

### run_subgoal: concurrent main loop

**Main thread loop (per step):**

1. `result = monitor.poll_result()` - check for completed VLM result
2. If `result` is not `None`:
   - `"force_converge"`: enter convergence sub-loop (see below)
   - `"continue"`: no action needed
3. Grab frame from camera, compute world pose and relative pose
4. `monitor.on_frame(frame_path, displacement)` - non-blocking in
   time-based mode
5. `openvla.predict()` - blocking ~0.2s
6. Send commands to hardware
7. Check convergence detection (small-delta pose check)
8. If converged: enter convergence sub-loop

**Convergence sub-loop:**

Entered when either `poll_result()` returns `"force_converge"` or the
small-delta convergence detector triggers.

1. `monitor.request_convergence(frame_path, displacement)` - queues the
   convergence VLM call
2. Loop: send zero-velocity `[0, 0, 0, 0]` commands at `command_dt_s`
   intervals
3. Each iteration: `result = monitor.poll_result()`
   - `"stop"`: subgoal complete, break the main loop
   - `"command"` with `new_instruction`: apply correction, reset OpenVLA
     origin, exit convergence sub-loop, resume normal commands
   - `None`: keep sending zero-velocity, continue waiting
4. Respect `stop_capture` flag for interrupt handling

**Convergence detection timing:**

The existing guard `steps_since_correction >= check_interval` changes to
a time-based guard in time-based mode: `time.time() - last_correction_time
>= check_interval_s`. The small-delta counting (`small_count >=
ACTION_SMALL_STEPS`) stays step-based since it measures consecutive
command steps.

### CLI and entry point changes

**`run_hardware.py` / `parse_args` in `interface.py`:**

New argument:

```
--diary_check_interval_s  float  default=1.0
    Time-based checkpoint interval in seconds. When set, the VLM monitor
    runs checkpoints concurrently on a timer instead of every N frames.
    Set to 0 to use frame-based mode (--diary_check_interval).
```

`run_subgoal` gets a new parameter `check_interval_s: Optional[float]`
passed through to `LiveDiaryMonitor`.

**Simulation entry points (unchanged):**

Do not pass `check_interval_s`. The monitor defaults to `None`
(frame-based synchronous mode). No code changes needed.

### Thread safety summary

All shared state lives inside `LiveDiaryMonitor`, protected by a single
`threading.Lock`:

| Field | Writer | Reader |
|---|---|---|
| `_latest_frame_path` | main thread (on_frame) | monitor thread |
| `_latest_displacement` | main thread (on_frame) | monitor thread |
| `_pending_result` | monitor thread | main thread (poll_result) |
| `_convergence_request` | main thread (request_convergence) | monitor thread |
| `_frame_paths` | main thread (on_frame) | monitor thread (grid sampling) |

No lock contention issues expected: the main thread writes are fast
(pointer updates), and the monitor thread holds the lock only briefly to
read/write these fields. The VLM calls themselves happen outside the lock.

### Cleanup

`monitor.cleanup()` sets `_stop_event`, joins the background thread
(with a timeout), then proceeds with existing cleanup (temp dir removal).
The `finally` block in `run_subgoal` already calls `monitor.cleanup()`.

### What does NOT change

- `LiveDiaryMonitor` frame-based mode (all existing behavior)
- `mock_server.py` and all simulation code
- `OpenVLAClient` API
- `DroneControlClient` API
- `PoseManager` and odometry providers
- `diary_summary.json` output format (vlm_rtts already tracked)
- Signal handling and interrupt resilience
