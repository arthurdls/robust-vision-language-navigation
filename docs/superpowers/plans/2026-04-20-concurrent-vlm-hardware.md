# Concurrent VLM Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VLM monitor checkpoints run concurrently with the hardware command loop so the drone receives commands at OpenVLA inference speed (~0.2s) without blocking on VLM calls (1-3s).

**Architecture:** Add a time-based async mode to `LiveDiaryMonitor` that spawns an internal background thread for VLM checkpoints. The main thread feeds frames via `on_frame()` (non-blocking) and polls for results via `poll_result()`. The `run_subgoal` loop in `interface.py` is restructured with a convergence sub-loop that sends zero-velocity commands while waiting for VLM responses.

**Tech Stack:** Python threading, `threading.Lock`, `threading.Event`, `threading.Thread`

**Spec:** `docs/superpowers/specs/2026-04-20-concurrent-vlm-hardware-design.md`

---

### Task 1: Add time-based frame storage to LiveDiaryMonitor

**Files:**
- Modify: `src/rvln/ai/diary_monitor.py:225-250` (constructor)
- Modify: `src/rvln/ai/diary_monitor.py:288-317` (`on_frame`)

This task adds the `check_interval_s` parameter, timestamped frame storage, and time-based `on_frame` gating, but does NOT yet add the background thread. The monitor will just accumulate frames without running async checkpoints.

- [ ] **Step 1: Add `check_interval_s` parameter and time-based state to constructor**

In `src/rvln/ai/diary_monitor.py`, modify `__init__`:

```python
def __init__(
    self,
    subgoal: str,
    check_interval: int,
    model: str = "gpt-4o",
    artifacts_dir: Optional[Path] = None,
    max_corrections: int = 15,
    check_interval_s: Optional[float] = None,
):
    self._subgoal = subgoal
    self._check_interval = check_interval
    self._model = model
    self._artifacts_dir = artifacts_dir
    self._max_corrections = max_corrections
    self._check_interval_s = check_interval_s
    self._time_based = check_interval_s is not None

    self._llm: BaseLLM = self._make_llm(model)
    self._frame_paths: List[Path] = []
    self._frame_timestamps: List[float] = []
    self._diary: List[str] = []
    self._step = 0
    self._corrections_used = 0
    self._parse_failures = 0
    self._vlm_calls = 0
    self._vlm_rtts: List[Dict[str, Any]] = []
    self._last_completion_pct: float = 0.0
    self._high_water_mark: float = 0.0
    self._last_displacement: List[float] = [0.0, 0.0, 0.0, 0.0]
    self._temp_dir: Optional[str] = None
```

Add `import threading` to the imports at the top of the file (alongside existing `import time`).

- [ ] **Step 2: Add `_sample_frames_by_time` helper method**

Add this method in the Internal helpers section of `LiveDiaryMonitor` (after `_format_displacement`):

```python
def _sample_frames_by_time(self) -> List[Path]:
    """Select frames closest to each check_interval_s boundary."""
    if not self._frame_timestamps:
        return []
    t0 = self._frame_timestamps[0]
    t_last = self._frame_timestamps[-1]
    interval = self._check_interval_s
    boundaries = []
    t = t0
    while t <= t_last:
        boundaries.append(t)
        t += interval
    if not boundaries or boundaries[-1] < t_last:
        boundaries.append(t_last)
    sampled: List[Path] = []
    ts_idx = 0
    for boundary in boundaries:
        while ts_idx < len(self._frame_timestamps) - 1 and \
              abs(self._frame_timestamps[ts_idx + 1] - boundary) <= \
              abs(self._frame_timestamps[ts_idx] - boundary):
            ts_idx += 1
        sampled.append(self._frame_paths[ts_idx])
    seen: set = set()
    deduped: List[Path] = []
    for p in sampled:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped
```

- [ ] **Step 3: Modify `on_frame` to handle time-based mode**

Replace the `on_frame` method to branch on `self._time_based`:

```python
def on_frame(
    self,
    frame_image_or_path: Union[np.ndarray, Path, str],
    displacement: Optional[List[float]] = None,
) -> DiaryCheckResult:
    path = self._save_frame(frame_image_or_path)
    self._frame_paths.append(path)
    self._frame_timestamps.append(time.time())
    self._step += 1

    if displacement is not None:
        self._last_displacement = list(displacement)

    if self._time_based:
        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning="",
            diary_entry="",
            completion_pct=self._last_completion_pct,
        )

    if self._step % self._check_interval != 0 or self._step < self._check_interval:
        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning="",
            diary_entry="",
            completion_pct=self._last_completion_pct,
        )

    return self._run_checkpoint()
```

- [ ] **Step 4: Verify syntax**

Run: `python3 -c "import py_compile; py_compile.compile('src/rvln/ai/diary_monitor.py', doraise=True)"`
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add src/rvln/ai/diary_monitor.py
git commit -m "Add time-based frame storage and on_frame gating to LiveDiaryMonitor"
```

---

### Task 2: Add background thread, poll_result, and request_convergence

**Files:**
- Modify: `src/rvln/ai/diary_monitor.py:225-250` (constructor)
- Modify: `src/rvln/ai/diary_monitor.py` (new methods in Public API section)
- Modify: `src/rvln/ai/diary_monitor.py` (new `_monitor_loop` in Internal helpers)
- Modify: `src/rvln/ai/diary_monitor.py:636-644` (cleanup)

- [ ] **Step 1: Add threading state to constructor**

Add the following fields at the end of `__init__`, after `self._temp_dir`:

```python
    self._lock = threading.Lock()
    self._stop_event = threading.Event()
    self._pending_result: Optional[DiaryCheckResult] = None
    self._convergence_request: Optional[Tuple[Path, List[float]]] = None
    self._last_checkpoint_time: float = time.time()
    self._monitor_thread: Optional[threading.Thread] = None

    if self._time_based:
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True,
        )
        self._monitor_thread.start()
```

Add `Tuple` to the `typing` imports at the top of the file (it is not currently imported).

- [ ] **Step 2: Make on_frame thread-safe in time-based mode**

In the `on_frame` method, the time-based branch needs to update shared state under the lock. Replace the `if self._time_based:` block:

```python
    if self._time_based:
        with self._lock:
            self._frame_paths.append(path)
            self._frame_timestamps.append(time.time())
            self._step += 1
            if displacement is not None:
                self._last_displacement = list(displacement)
        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning="",
            diary_entry="",
            completion_pct=self._last_completion_pct,
        )
```

And move the non-time-based append/step-increment to happen before the time_based check (remove duplicates). The full revised `on_frame`:

```python
def on_frame(
    self,
    frame_image_or_path: Union[np.ndarray, Path, str],
    displacement: Optional[List[float]] = None,
) -> DiaryCheckResult:
    path = self._save_frame(frame_image_or_path)

    if self._time_based:
        with self._lock:
            self._frame_paths.append(path)
            self._frame_timestamps.append(time.time())
            self._step += 1
            if displacement is not None:
                self._last_displacement = list(displacement)
        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning="",
            diary_entry="",
            completion_pct=self._last_completion_pct,
        )

    self._frame_paths.append(path)
    self._frame_timestamps.append(time.time())
    self._step += 1

    if displacement is not None:
        self._last_displacement = list(displacement)

    if self._step % self._check_interval != 0 or self._step < self._check_interval:
        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning="",
            diary_entry="",
            completion_pct=self._last_completion_pct,
        )

    return self._run_checkpoint()
```

- [ ] **Step 3: Add `poll_result` and `request_convergence` to Public API**

Add these after the `vlm_rtts` property:

```python
def poll_result(self) -> Optional[DiaryCheckResult]:
    """Return and clear any pending async result.
    Returns None if no result is ready or if in frame-based mode."""
    if not self._time_based:
        return None
    with self._lock:
        result = self._pending_result
        self._pending_result = None
    return result

def request_convergence(
    self, frame_path: Union[Path, str], displacement: List[float],
) -> None:
    """Queue a convergence check for the background thread."""
    with self._lock:
        self._convergence_request = (Path(frame_path), list(displacement))
```

- [ ] **Step 4: Add `_run_checkpoint_async` helper**

The existing `_run_checkpoint` accesses `self._frame_paths` and `self._step` directly. For the background thread, we need a version that takes a snapshot of the data under the lock and then runs without holding it. Add this method in the Internal helpers section:

```python
def _run_checkpoint_async(self) -> Optional[DiaryCheckResult]:
    """Run a checkpoint from the background thread using time-sampled frames."""
    with self._lock:
        if len(self._frame_paths) < 2:
            return None
        step = self._step
        prev_path = self._frame_paths[-2] if len(self._frame_paths) >= 2 else self._frame_paths[-1]
        curr_path = self._frame_paths[-1]
        displacement = list(self._last_displacement)
        all_frame_paths = list(self._frame_paths)
        all_frame_timestamps = list(self._frame_timestamps)

    self._last_displacement = displacement

    grid_two = build_frame_grid([prev_path, curr_path])
    prompt_local = LOCAL_PROMPT_TEMPLATE.format(subgoal=self._subgoal)
    change_text = self._timed_query_vlm(
        grid_two, prompt_local, "local_checkpoint",
        system_prompt=GENERAL_SYSTEM_PROMPT,
    )
    disp_str = self._format_displacement()
    diary_entry = f"Steps ~{step} {disp_str}: {change_text}"
    self._diary.append(diary_entry)

    sampled = self._sample_frames_by_time()
    if not sampled:
        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning="No sampled frames for global grid.",
            diary_entry=diary_entry,
            completion_pct=self._last_completion_pct,
        )

    sampled = sampled[-self.MAX_GLOBAL_FRAMES:]

    grid_global = build_frame_grid(sampled)
    diary_blob = "\n".join(self._diary)
    prompt_global = GLOBAL_PROMPT_TEMPLATE.format(
        subgoal=self._subgoal,
        diary=diary_blob,
        prev_completion_pct=self._last_completion_pct,
        displacement=disp_str,
    )

    response_global = self._timed_query_vlm(
        grid_global, prompt_global, "global_checkpoint",
        system_prompt=GENERAL_SYSTEM_PROMPT,
    )

    parsed = self._parse_json_response(response_global)
    if not parsed:
        self._parse_failures += 1
        logger.warning(
            "Checkpoint %d JSON parse failed, retrying. Raw: %s",
            step, response_global[:200],
        )
        response_global = self._timed_query_vlm(
            grid_global, prompt_global, "global_checkpoint_retry",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        parsed = self._parse_json_response(response_global)
        if not parsed:
            self._parse_failures += 1
            logger.error(
                "Checkpoint %d JSON parse failed after retry. Raw: %s",
                step, response_global[:200],
            )

    self._save_checkpoint_artifact(
        step, grid_two, grid_global,
        prompt_local, change_text,
        prompt_global, response_global,
    )

    result = self._parse_global_response(
        response_global, diary_entry, parsed=parsed,
    )

    self._last_completion_pct = result.completion_pct
    self._high_water_mark = max(self._high_water_mark, result.completion_pct)
    self._diary.append(
        f"Checkpoint {step}: completion = {result.completion_pct:.2f}"
    )

    return result
```

- [ ] **Step 5: Add `_run_convergence_async` helper**

Similar to `_run_checkpoint_async`, this runs the convergence VLM call from the background thread. Add after `_run_checkpoint_async`:

```python
def _run_convergence_async(
    self, frame_path: Path, displacement: List[float],
) -> DiaryCheckResult:
    """Run convergence check from the background thread."""
    with self._lock:
        if frame_path not in self._frame_paths or self._frame_paths[-1] != frame_path:
            self._frame_paths.append(frame_path)
            self._frame_timestamps.append(time.time())
        self._last_displacement = list(displacement)

    if self._corrections_used >= self._max_corrections:
        logger.warning(
            "Max corrections (%d) exhausted. Ending run.",
            self._max_corrections,
        )
        return DiaryCheckResult(
            action="stop",
            new_instruction="",
            reasoning=f"Max corrections ({self._max_corrections}) exhausted.",
            diary_entry="",
            completion_pct=self._last_completion_pct,
        )

    disp_str = self._format_displacement()
    diary_blob = "\n".join(self._diary) if self._diary else "(no diary entries yet)"
    prompt = CONVERGENCE_PROMPT_TEMPLATE.format(
        subgoal=self._subgoal,
        diary=diary_blob,
        prev_completion_pct=self._last_completion_pct,
        displacement=disp_str,
    )

    sampled = self._sample_frames_by_time()
    if not sampled or sampled[-1] != frame_path:
        sampled.append(frame_path)
    sampled = sampled[-self.MAX_GLOBAL_FRAMES:]

    grid = build_frame_grid(sampled)
    response = self._timed_query_vlm(
        grid, prompt, "convergence",
        system_prompt=GENERAL_SYSTEM_PROMPT,
    )

    self._save_convergence_artifact(response, prompt, grid)

    parsed = self._parse_json_response(response)
    if not parsed:
        self._parse_failures += 1
        logger.warning(
            "Convergence JSON parse failed (attempt 1), retrying. Raw: %s",
            response[:200],
        )
        response = self._timed_query_vlm(
            grid, prompt, "convergence_retry",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        self._save_convergence_artifact(response, prompt, grid)
        parsed = self._parse_json_response(response)
        if not parsed:
            self._parse_failures += 1
            logger.error(
                "Convergence JSON parse failed after retry. Raw: %s",
                response[:200],
            )

    pct = float(parsed.get("completion_percentage", self._last_completion_pct))
    pct = max(0.0, min(1.0, pct))
    self._last_completion_pct = pct
    self._high_water_mark = max(self._high_water_mark, pct)

    if parsed.get("complete", False) or parsed.get("diagnosis") == "complete":
        return DiaryCheckResult(
            action="stop",
            new_instruction="",
            reasoning=f"Subgoal complete on convergence. Raw: {response}",
            diary_entry="",
            completion_pct=pct,
        )

    corrective = parsed.get("corrective_instruction") or ""
    if not corrective:
        logger.warning(
            "Convergence response missing corrective_instruction, retrying."
        )
        response = self._timed_query_vlm(
            grid, prompt, "convergence_instruction_retry",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        self._save_convergence_artifact(response, prompt, grid)
        parsed_retry = self._parse_json_response(response)
        if parsed_retry.get("complete", False) or parsed_retry.get("diagnosis") == "complete":
            pct_r = float(parsed_retry.get("completion_percentage", pct))
            pct_r = max(0.0, min(1.0, pct_r))
            self._last_completion_pct = pct_r
            self._high_water_mark = max(self._high_water_mark, pct_r)
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning=f"Subgoal complete on convergence (retry). Raw: {response}",
                diary_entry="",
                completion_pct=pct_r,
            )
        corrective = (parsed_retry.get("corrective_instruction") or "").strip()
        if not corrective:
            logger.error("Convergence retry also returned no instruction.")
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning="no_corrective_instruction",
                diary_entry="",
                completion_pct=pct,
            )

    self._corrections_used += 1
    return DiaryCheckResult(
        action="command",
        new_instruction=corrective,
        reasoning=f"Convergence diagnosis: {parsed.get('diagnosis', 'unknown')}. Raw: {response}",
        diary_entry="",
        completion_pct=pct,
    )
```

- [ ] **Step 6: Add `_monitor_loop` method**

Add this in the Internal helpers section:

```python
def _monitor_loop(self) -> None:
    """Background thread: runs checkpoints on a timer, handles convergence requests."""
    while not self._stop_event.is_set():
        self._stop_event.wait(timeout=0.05)
        if self._stop_event.is_set():
            break

        with self._lock:
            conv_req = self._convergence_request
            if conv_req is not None:
                self._convergence_request = None

        if conv_req is not None:
            frame_path, displacement = conv_req
            try:
                result = self._run_convergence_async(frame_path, displacement)
            except Exception as exc:
                logger.error("Async convergence failed: %s", exc)
                result = DiaryCheckResult(
                    action="stop",
                    new_instruction="",
                    reasoning=f"convergence_error: {exc}",
                    diary_entry="",
                    completion_pct=self._last_completion_pct,
                )
            with self._lock:
                self._pending_result = result
                self._last_checkpoint_time = time.time()
            continue

        elapsed = time.time() - self._last_checkpoint_time
        if elapsed < self._check_interval_s:
            continue

        try:
            result = self._run_checkpoint_async()
        except Exception as exc:
            logger.error("Async checkpoint failed: %s", exc)
            result = None

        if result is not None:
            with self._lock:
                self._pending_result = result
        self._last_checkpoint_time = time.time()
```

- [ ] **Step 7: Update `cleanup` to stop the background thread**

Replace the existing `cleanup` method:

```python
def cleanup(self) -> None:
    """Stop background thread and remove temporary frame directory."""
    self._stop_event.set()
    if self._monitor_thread is not None:
        self._monitor_thread.join(timeout=5.0)
        self._monitor_thread = None
    if self._temp_dir is not None:
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
        self._temp_dir = None
```

- [ ] **Step 8: Verify syntax**

Run: `python3 -c "import py_compile; py_compile.compile('src/rvln/ai/diary_monitor.py', doraise=True)"`
Expected: no output (success)

- [ ] **Step 9: Commit**

```bash
git add src/rvln/ai/diary_monitor.py
git commit -m "Add background thread, poll_result, and request_convergence to LiveDiaryMonitor"
```

---

### Task 3: Restructure run_subgoal for concurrent execution

**Files:**
- Modify: `src/rvln/mininav/interface.py:562-770` (`run_subgoal`)

- [ ] **Step 1: Add `check_interval_s` parameter to `run_subgoal`**

Add `check_interval_s: Optional[float] = None,` to the function signature after `action_pose_mode`:

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
) -> Dict[str, Any]:
```

- [ ] **Step 2: Pass `check_interval_s` to LiveDiaryMonitor**

In the `run_subgoal` function body, update the monitor construction:

```python
    monitor = LiveDiaryMonitor(
        subgoal=subgoal_nl,
        check_interval=check_interval,
        model=monitor_model,
        artifacts_dir=diary_artifacts,
        max_corrections=max_corrections,
        check_interval_s=check_interval_s,
    )
```

- [ ] **Step 3: Add convergence sub-loop helper**

Add this helper function BEFORE `run_subgoal` (at module level in `interface.py`):

```python
def _convergence_loop(
    monitor,
    control: DroneControlClient,
    pose_manager: PoseManager,
    camera,
    frames_dir: Path,
    frame_offset: int,
    step: int,
    command_dt_s: float,
    origin_world: List[float],
    trajectory_log: List[Dict[str, Any]],
    override_history: List[Dict[str, Any]],
    current_instruction: str,
    subgoal_rel_pose: List[float],
    frame_path: Path,
) -> Optional[Dict[str, Any]]:
    """Send zero-velocity commands while waiting for convergence VLM result.

    Returns a dict with keys 'action', 'new_instruction', 'reasoning',
    'completion_pct' when the monitor responds, or None if interrupted.
    """
    monitor.request_convergence(frame_path, list(subgoal_rel_pose))
    zero_cmd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    global_frame_idx = frame_offset + step

    while not stop_capture:
        result = monitor.poll_result()
        if result is not None:
            return {
                "action": result.action,
                "new_instruction": result.new_instruction,
                "reasoning": result.reasoning,
                "completion_pct": result.completion_pct,
            }
        control.send_command(global_frame_idx, zero_cmd)
        pose_manager.update_from_command(zero_cmd, command_dt_s)
        time.sleep(command_dt_s)

    return None
```

- [ ] **Step 4: Restructure the main loop for time-based mode**

Replace the main loop body inside `run_subgoal` (lines 609-739). The new loop handles both modes:

```python
    use_async = check_interval_s is not None
    last_correction_time = time.time() if use_async else None
    subgoal_rel_pose = [0.0, 0.0, 0.0, 0.0]
    frame_path = None

    try:
      for step in range(max_steps):
        if stop_capture:
            stop_reason = "interrupted"
            break

        if use_async:
            async_result = monitor.poll_result()
            if async_result is not None and async_result.action == "force_converge":
                conv_out = _convergence_loop(
                    monitor=monitor,
                    control=control,
                    pose_manager=pose_manager,
                    camera=camera,
                    frames_dir=frames_dir,
                    frame_offset=frame_offset,
                    step=step,
                    command_dt_s=command_dt_s,
                    origin_world=origin_world,
                    trajectory_log=trajectory_log,
                    override_history=override_history,
                    current_instruction=current_instruction,
                    subgoal_rel_pose=subgoal_rel_pose,
                    frame_path=frame_path or (frames_dir / f"frame_{frame_offset + step:06d}.png"),
                )
                if conv_out is None:
                    stop_reason = "interrupted"
                    break
                if conv_out["action"] == "stop":
                    stop_reason = "monitor_complete"
                    total_steps = step
                    break
                if conv_out.get("new_instruction"):
                    override_history.append({
                        "step": step,
                        "type": f"convergence_{conv_out['action']}",
                        "old_instruction": current_instruction,
                        "new_instruction": conv_out["new_instruction"],
                        "reasoning": conv_out["reasoning"],
                    })
                    current_instruction = conv_out["new_instruction"]
                    openvla_pose_origin = list(relative_pose(
                        pose_manager.get_world_pose(), origin_world
                    ))
                    small_count = 0
                    last_pose = None
                    last_correction_time = time.time()
                    last_correction_step = step
                    openvla.reset_model()
                else:
                    stop_reason = "convergence_no_command"
                    break

        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        world_pose = pose_manager.get_world_pose()
        subgoal_rel_pose = relative_pose(world_pose, origin_world)

        global_frame_idx = frame_offset + step
        frame_path = frames_dir / f"frame_{global_frame_idx:06d}.png"
        cv2.imwrite(str(frame_path), frame)

        if use_async:
            monitor.on_frame(frame_path, displacement=list(subgoal_rel_pose))
        else:
            try:
                result = monitor.on_frame(frame_path, displacement=list(subgoal_rel_pose))
            except Exception as exc:
                logger.error("monitor.on_frame failed at step %d: %s", step, exc)
                result = DiaryCheckResult(
                    action="continue",
                    new_instruction="",
                    reasoning="monitor_error",
                    diary_entry="",
                    completion_pct=monitor.last_completion_pct,
                )

            if result.action == "stop":
                stop_reason = "monitor_complete"
                total_steps = step
                break

            if result.action == "force_converge":
                override_history.append({
                    "step": step,
                    "type": "force_converge",
                    "reasoning": result.reasoning,
                })

        openvla_pose = [c - o for c, o in zip(subgoal_rel_pose, openvla_pose_origin)]
        response = openvla.predict(
            image_bgr=frame,
            proprio=state_for_openvla(openvla_pose),
            instr=current_instruction.strip().lower(),
        )
        if response is None:
            stop_reason = "no_response"
            total_steps = step
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            stop_reason = "empty_action"
            total_steps = step
            break

        for action_pose in action_poses:
            if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 4):
                continue
            current_world = pose_manager.get_world_pose()
            current_rel = relative_pose(current_world, origin_world)
            cmd = to_command_from_action_pose(action_pose, current_rel, action_pose_mode)
            control.send_command(global_frame_idx, cmd)
            pose_manager.update_from_command(cmd, command_dt_s)
            updated_rel = relative_pose(pose_manager.get_world_pose(), origin_world)
            trajectory_log.append({
                "state": [
                    [updated_rel[0], updated_rel[1], updated_rel[2]],
                    [0, updated_rel[3], 0],
                ]
            })
            time.sleep(command_dt_s)

        total_steps = step + 1
        world_pose = pose_manager.get_world_pose()
        subgoal_rel_pose = relative_pose(world_pose, origin_world)

        if use_async:
            converged = False
            time_since_correction = time.time() - last_correction_time
            if last_pose is not None and time_since_correction >= check_interval_s:
                diffs = [abs(a - b) for a, b in zip(subgoal_rel_pose, last_pose)]
                if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                    small_count += 1
                else:
                    small_count = 0
                if small_count >= ACTION_SMALL_STEPS:
                    converged = True
            last_pose = list(subgoal_rel_pose)

            if converged:
                conv_out = _convergence_loop(
                    monitor=monitor,
                    control=control,
                    pose_manager=pose_manager,
                    camera=camera,
                    frames_dir=frames_dir,
                    frame_offset=frame_offset,
                    step=step,
                    command_dt_s=command_dt_s,
                    origin_world=origin_world,
                    trajectory_log=trajectory_log,
                    override_history=override_history,
                    current_instruction=current_instruction,
                    subgoal_rel_pose=subgoal_rel_pose,
                    frame_path=frame_path,
                )
                if conv_out is None:
                    stop_reason = "interrupted"
                    break
                if conv_out["action"] == "stop":
                    stop_reason = "monitor_complete"
                    total_steps = step
                    break
                if conv_out.get("new_instruction"):
                    override_history.append({
                        "step": step,
                        "type": f"convergence_{conv_out['action']}",
                        "old_instruction": current_instruction,
                        "new_instruction": conv_out["new_instruction"],
                        "reasoning": conv_out["reasoning"],
                    })
                    current_instruction = conv_out["new_instruction"]
                    openvla_pose_origin = list(subgoal_rel_pose)
                    small_count = 0
                    last_pose = None
                    last_correction_time = time.time()
                    last_correction_step = step
                    openvla.reset_model()
                else:
                    stop_reason = "convergence_no_command"
                    break
        else:
            converged = result.action == "force_converge"
            steps_since_correction = step - last_correction_step

            if last_pose is not None and steps_since_correction >= check_interval:
                diffs = [abs(a - b) for a, b in zip(subgoal_rel_pose, last_pose)]
                if all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3]) and diffs[3] < ACTION_SMALL_DELTA_YAW:
                    small_count += 1
                else:
                    small_count = 0
                if small_count >= ACTION_SMALL_STEPS:
                    converged = True
            last_pose = list(subgoal_rel_pose)

            if converged:
                try:
                    conv_result = monitor.on_convergence(
                        frame_path, displacement=list(subgoal_rel_pose)
                    )
                except Exception as exc:
                    logger.error("monitor.on_convergence failed at step %d: %s", step, exc)
                    conv_result = DiaryCheckResult(
                        action="stop",
                        new_instruction="",
                        reasoning="convergence_monitor_error",
                        diary_entry="",
                        completion_pct=monitor.last_completion_pct,
                    )

                if conv_result.action == "stop":
                    stop_reason = "monitor_complete"
                    break

                if conv_result.new_instruction:
                    override_history.append({
                        "step": step,
                        "type": f"convergence_{conv_result.action}",
                        "old_instruction": current_instruction,
                        "new_instruction": conv_result.new_instruction,
                        "reasoning": conv_result.reasoning,
                    })
                    current_instruction = conv_result.new_instruction
                    openvla_pose_origin = list(subgoal_rel_pose)
                    small_count = 0
                    last_pose = None
                    last_correction_step = step
                    openvla.reset_model()
                else:
                    stop_reason = "convergence_no_command"
                    break
      else:
          stop_reason = "max_steps"
          total_steps = max_steps
    except Exception as exc:
        stop_reason = f"error: {exc}"
        logger.error("run_subgoal failed at step %d: %s", total_steps, exc)
```

- [ ] **Step 5: Verify syntax**

Run: `python3 -c "import py_compile; py_compile.compile('src/rvln/mininav/interface.py', doraise=True)"`
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add src/rvln/mininav/interface.py
git commit -m "Restructure run_subgoal for concurrent VLM execution"
```

---

### Task 4: Add CLI argument and wire through main()

**Files:**
- Modify: `src/rvln/mininav/interface.py:772-840` (`parse_args`)
- Modify: `src/rvln/mininav/interface.py:842-1016` (`main`)

- [ ] **Step 1: Add `--diary_check_interval_s` CLI argument**

In `parse_args`, add after the `--diary_check_interval` argument:

```python
    parser.add_argument(
        "--diary_check_interval_s",
        type=float,
        default=1.0,
        help=(
            "Time-based checkpoint interval in seconds for concurrent VLM "
            "monitoring. Default: 1.0. Set to 0 to use frame-based mode "
            "(--diary_check_interval)."
        ),
    )
```

- [ ] **Step 2: Wire `check_interval_s` through to `run_subgoal` in `main()`**

In `main()`, compute the value and pass it to `run_subgoal`. Add before the subgoal while loop:

```python
        check_interval_s = args.diary_check_interval_s if args.diary_check_interval_s > 0 else None
```

Then add `check_interval_s=check_interval_s,` to the `run_subgoal(...)` call (after `action_pose_mode=args.action_pose_mode,`):

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
            )
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -c "import py_compile; py_compile.compile('src/rvln/mininav/interface.py', doraise=True)"`
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add src/rvln/mininav/interface.py
git commit -m "Add --diary_check_interval_s CLI arg for concurrent VLM mode"
```

---

### Task 5: Verify end-to-end and push

**Files:**
- All modified files

- [ ] **Step 1: Verify both files compile**

Run:
```bash
python3 -c "import py_compile; py_compile.compile('src/rvln/ai/diary_monitor.py', doraise=True)"
python3 -c "import py_compile; py_compile.compile('src/rvln/mininav/interface.py', doraise=True)"
```
Expected: no output (success) for both

- [ ] **Step 2: Verify the CLI help shows the new argument**

Run: `python3 scripts/run_hardware.py --help 2>&1 | grep diary_check_interval_s`
Expected: shows the `--diary_check_interval_s` argument with help text

- [ ] **Step 3: Run existing tests to check for regressions**

Run: `python3 -m pytest tests/ -v 2>&1 | tail -20`
Expected: existing tests pass (test_ltl_planner.py)

- [ ] **Step 4: Push all commits**

Run: `git push`
