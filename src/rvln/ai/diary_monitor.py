"""
Live diary-based subgoal completion monitor with supervisor capabilities.

Every N steps during normal execution: builds a local 2-frame grid
(what changed?) and a global sampled grid (is the subgoal complete?
is the drone on-track?), tracks estimated completion percentage and
displacement, and returns an action: continue / stop / override /
command / ask_help / force_converge.

On convergence (drone stops): evaluates whether the subgoal is truly
complete or whether corrective commands should be issued to OpenVLA.
If the correction budget is exhausted, returns ask_help so the runner
can prompt the human operator.

Stall detection: tracks completion percentage history across checkpoints.
If completion plateaus (spread < stall_threshold over the last
stall_window checkpoints), returns ask_help instead of continue.
Stall detection is suppressed when completion is already high
(above stall_completion_floor).

Key behaviours:
  - Completion percentage is reported as-is from the LLM (no clamping).
    A separate peak_completion value tracks the maximum reached.
  - The global prompt asks the LLM for a ``should_override`` flag (rather
    than separate overshot / off-track flags). When true and a
    corrective_instruction is provided, the runner replaces the current
    OpenVLA instruction.
  - The system prompt emphasises completion strictness (never mark complete
    unless highly confident, cap at 0.95 when uncertain) and proactive
    correction (flag overshoot early, prefer small corrections over large
    late ones).
"""

import json
import logging
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .utils.llm_providers import BaseLLM, LLMFactory
from .utils.vision import build_frame_grid, query_vlm, sample_frames_every_n

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiaryCheckResult:
    action: str           # "continue", "stop", "override", "command", "ask_help", or "force_converge"
    new_instruction: str  # populated when action is "override" or "command"
    reasoning: str
    diary_entry: str      # latest diary entry (what changed)
    completion_pct: float = 0.0  # latest estimated completion percentage


# ---------------------------------------------------------------------------
# System prompts (general -- not per-task)
# ---------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = """\
You are a completion monitor for an autonomous drone executing a single subgoal.
You watch first-person video frames and a running diary to decide whether the
subgoal is done, and issue corrections when the drone stops prematurely.

COMPLETION CRITERIA — mark complete only with high confidence:
- MOVEMENT ("move past X"): drone has passed the landmark.
- BETWEEN ("go between X and Y"): drone is positioned between both landmarks.
- APPROACH ("approach X"): target fills a large portion of the frame.
- VISUAL SEARCH ("turn until you see X"): target is clearly visible in the
  frame. It does NOT need to be perfectly centered — anywhere in the frame is
  acceptable as long as it is identifiable.
- ABOVE ("go above X"): target is visible below — requires being positioned
  over the target, not just at a higher altitude.
- BELOW ("go below X"): target is visible above.
- TRAVERSAL ("move through X"): drone has passed through the structure.
Never set completion_percentage to 1.0 unless certain. Cap at 0.95 when unsure.

DISPLACEMENT: [x, y, z, yaw] relative to subtask start. x/y are fixed to the
initial heading (x = forward, y = lateral at start). z = altitude. Meters.
yaw = heading change in degrees.

DURING NORMAL FLIGHT — your primary job is to detect completion and problems:
- If the subgoal is complete, set "complete" to true.
- If the drone is actively making things worse (e.g., moving away from the target,
  overshooting), set "should_stop" to true so it can be corrected.
- Otherwise, let the drone execute its instruction without interference.

ORIENTATION TOLERANCE — avoid oscillating corrections:
- When the subgoal involves turning toward or facing an object, the target does
  NOT need to be at the exact center of the frame. If the target is visible
  anywhere in the frame, the orientation is good enough — mark it complete
  rather than issuing further yaw corrections.

WHEN THE DRONE STOPS (convergence corrections):
- Decide if the subgoal is complete, stopped short, or overshot.
- Issue ONE single-action corrective command — the drone cannot execute compound
  instructions like "ascend and move forward". Pick the single axis that is the
  biggest bottleneck right now; the other axes will be addressed in subsequent
  correction cycles. The diary highlights the most visually obvious changes,
  which may not reflect the real bottleneck — check displacement data and think
  about what the subgoal actually requires.
- Retreat commands must reference the target object (e.g., "move back from the
  [object]") — the drone does not understand bare "move backward X meters".
- Keep corrections small (under 1.0 meters) for frequent re-evaluation."""

LOCAL_PROMPT_TEMPLATE = """\
The subgoal is: {subgoal}

What changed between these two consecutive frames relative to this subgoal?
Answer in ONE short sentence with only the key facts that directly bear on the subgoal."""

GLOBAL_PROMPT_TEMPLATE = """\
Subgoal: {subgoal}

Previous estimated completion: {prev_completion_pct}
Current displacement from start: [x, y, z, yaw] = {displacement}

Diary of changes observed so far:
{diary}

The grid shows up to the 9 most recent sampled frames (left to right, top to
bottom, in temporal order). If there are more than 9 diary entries, earlier frames
are no longer visible in the grid — rely on the diary text for that history.

Based on the diary and the grid of sampled frames, respond with EXACTLY ONE JSON
object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "on_track": true/false,
  "should_stop": true/false
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished. Do NOT mark complete for partial progress.
- "completion_percentage": your best estimate of how close the subgoal is to
  completion (0.0 = not started, 1.0 = fully done). NEVER set 1.0 unless you
  are highly confident — use at most 0.95 when unsure.
- "on_track": true if the drone is making any progress toward the subgoal.
- "should_stop": true only if the drone is actively making things worse (e.g.,
  overshooting, moving away from target). The drone will be stopped and a
  correction issued. Do NOT set true for slow progress."""

CONVERGENCE_PROMPT_TEMPLATE = """\
Subgoal: {subgoal}

Previous estimated completion: {prev_completion_pct}
Current displacement from start: [x, y, z, yaw] = {displacement}

Diary of changes observed so far:
{diary}

The drone has stopped moving. The grid shows up to the 9 most recent sampled
frames (left to right, top to bottom, in temporal order). If there are more
than 9 diary entries, earlier frames are no longer visible in the grid — rely
on the diary text for that history.

Given the diary and the sampled frames, is the subgoal complete? If not, did
the drone stop short or overshoot?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "completion_percentage": 0.0 to 1.0,
  "diagnosis": "stopped_short" or "overshot" or "complete",
  "corrective_instruction": "..." or null
}}

- "complete": true ONLY if you are highly confident the subgoal has been fully
  accomplished. Do NOT mark complete for partial progress. When in doubt, keep
  it false and issue a corrective instruction.
- "completion_percentage": your best estimate of how close the subgoal is to
  completion (0.0 = not started, 1.0 = fully done). NEVER set 1.0 unless you
  are highly confident the subtask is fully complete — use at most 0.95 if the
  result looks close but you are not certain.
- "diagnosis": "complete" if done, "stopped_short" if the drone needs to keep
  going, "overshot" if the drone went past the goal.
- "corrective_instruction": REQUIRED if not complete — a single-action drone
  command to fix the biggest gap (not compound — one action per correction).
  null only if complete.

  Useful corrective patterns:
    * "Turn toward <landmark>" — re-orient the drone toward a visible or
      expected landmark so the policy can locate it.
    * "Turn right/left <N> degrees" — precise yaw adjustment when the target
      is off-screen or partially visible.
    * "Move forward <N> meters" / "Move closer to <landmark>" — close a gap.
    * "Ascend/Descend <N> meters" — altitude correction.
  Prefer a turn command when the target is not visible in the latest frame;
  the underlying policy needs to see the target to navigate toward it.

  IMPORTANT — orientation tolerance: if the subgoal is about turning toward or
  facing a target and the target is already visible in the frame (even if
  off-center), mark the subgoal complete instead of issuing further turn
  corrections. Small yaw offsets are acceptable. Do NOT oscillate between
  left and right turn corrections trying to perfectly center the target."""



# ---------------------------------------------------------------------------
# LiveDiaryMonitor
# ---------------------------------------------------------------------------

class LiveDiaryMonitor:
    """Real-time diary-based subgoal completion monitor with supervisor capabilities.

    Designed to monitor a SINGLE subgoal (one predicate from the LTL planner).
    Operates in three modes:

    1. PASSIVE MONITORING (during normal execution):
       Every N steps (or N seconds in async mode): builds local 2-frame grid,
       builds global sampled grid (capped at 9 frames for a 3x3 layout),
       maintains running diary with displacement and completion percentage,
       and returns continue / stop / force_converge / ask_help.

    2. SUPERVISOR MODE (when drone converges / stops prematurely):
       When the control loop detects convergence but the subgoal is not complete,
       the monitor issues corrective commands directly to OpenVLA until the
       subgoal is achieved or the correction budget is exhausted.

    3. STALL DETECTION (during passive monitoring):
       Tracks completion percentage history across checkpoints. If completion
       plateaus over the last ``stall_window`` checkpoints (spread below
       ``stall_threshold``), returns ask_help so the runner can prompt the
       human operator. Suppressed when completion exceeds
       ``stall_completion_floor``. When the correction budget is exhausted
       during supervisor mode, also returns ask_help instead of stopping.

    Parameters
    ----------
    subgoal : str
        Natural-language description of the current subgoal.
    check_interval : int
        Run a checkpoint every N frames (sync mode).
    model : str
        LLM model name for VLM queries.
    artifacts_dir : Path, optional
        Directory to save checkpoint and convergence artifacts.
    max_corrections : int
        Maximum corrective commands before asking for help.
    check_interval_s : float, optional
        If set, enables async (time-based) mode with checkpoints every N seconds.
    stall_window : int
        Number of consecutive checkpoints to consider for stall detection.
    stall_threshold : float
        Maximum spread (max - min) of completion across the stall window
        to count as stalled.
    stall_completion_floor : float
        Suppress stall detection when the minimum recent completion
        exceeds this value.
    """

    MAX_GLOBAL_FRAMES = 9

    def __init__(
        self,
        subgoal: str,
        check_interval: int,
        model: str = "gpt-4o",
        artifacts_dir: Optional[Path] = None,
        max_corrections: int = 15,
        check_interval_s: Optional[float] = None,
        stall_window: int = 3,
        stall_threshold: float = 0.05,
        stall_completion_floor: float = 0.8,
    ):
        self._subgoal = subgoal
        self._check_interval = check_interval
        self._model = model
        self._artifacts_dir = artifacts_dir
        self._max_corrections = max_corrections

        # Time-based mode configuration
        self._time_based: bool = check_interval_s is not None
        self._check_interval_s: Optional[float] = check_interval_s

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
        self._peak_completion: float = 0.0
        self._last_displacement: List[float] = [0.0, 0.0, 0.0, 0.0]
        self._stall_window = stall_window
        self._stall_threshold = stall_threshold
        self._stall_completion_floor = stall_completion_floor
        self._completion_history: List[float] = []
        self._temp_dir: Optional[str] = None

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def diary(self) -> List[str]:
        return list(self._diary)

    @property
    def step(self) -> int:
        return self._step

    @property
    def corrections_used(self) -> int:
        return self._corrections_used

    @property
    def last_completion_pct(self) -> float:
        return self._last_completion_pct

    @property
    def peak_completion(self) -> float:
        return self._peak_completion

    @property
    def parse_failures(self) -> int:
        return self._parse_failures

    @property
    def vlm_calls(self) -> int:
        return self._vlm_calls

    @property
    def vlm_rtts(self) -> List[Dict[str, Any]]:
        return list(self._vlm_rtts)

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

    def on_frame(
        self,
        frame_image_or_path: Union[np.ndarray, Path, str],
        displacement: Optional[List[float]] = None,
    ) -> DiaryCheckResult:
        """Process one frame during normal execution.

        On non-checkpoint steps returns action="continue" with empty diary_entry.
        On checkpoint steps (every check_interval) runs two LLM queries and
        returns the assessed action.

        Parameters
        ----------
        frame_image_or_path : frame as numpy array or path to saved image
        displacement : [x, y, z, yaw] relative to subtask start position
        """
        path = self._save_frame(frame_image_or_path)

        # In time-based mode, frame data updates must be under the lock
        # because the background thread reads them concurrently.
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

    def on_convergence(
        self,
        latest_frame: Union[np.ndarray, Path, str],
        displacement: Optional[List[float]] = None,
    ) -> DiaryCheckResult:
        """Called when the control loop detects the drone has stopped (convergence).

        Evaluates whether the subgoal is truly complete or if corrective commands
        are needed. Returns 'stop' if complete, 'command' with a corrective
        instruction if not, or 'ask_help' if the correction budget is exhausted.
        """
        path = self._save_frame(latest_frame)
        if not self._frame_paths or self._frame_paths[-1] != path:
            self._frame_paths.append(path)

        if displacement is not None:
            self._last_displacement = list(displacement)

        if self._corrections_used >= self._max_corrections:
            logger.warning(
                "Max corrections (%d) exhausted. Asking for help.",
                self._max_corrections,
            )
            return DiaryCheckResult(
                action="ask_help",
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

        sampled = sample_frames_every_n(self._frame_paths, self._check_interval)
        if not sampled or sampled[-1] != path:
            sampled.append(path)
        sampled = sampled[-self.MAX_GLOBAL_FRAMES:]

        grid = build_frame_grid(sampled)
        response = self._timed_query_vlm(
            grid, prompt, "convergence",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )

        self._save_convergence_artifact(response, prompt, grid)

        parsed = self._parse_json_response(response)
        if parsed is None:
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
            if parsed is None:
                self._parse_failures += 1
                raise RuntimeError(
                    f"Convergence JSON parse failed after retry. Raw: {response[:200]}"
                )

        pct = float(parsed.get("completion_percentage", self._last_completion_pct))
        pct = max(0.0, min(1.0, pct))
        self._last_completion_pct = pct
        self._peak_completion = max(self._peak_completion, pct)

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
            if parsed_retry is not None and (
                parsed_retry.get("complete", False) or parsed_retry.get("diagnosis") == "complete"
            ):
                pct_r = float(parsed_retry.get("completion_percentage", pct))
                pct_r = max(0.0, min(1.0, pct_r))
                self._last_completion_pct = pct_r
                self._peak_completion = max(self._peak_completion, pct_r)
                return DiaryCheckResult(
                    action="stop",
                    new_instruction="",
                    reasoning=f"Subgoal complete on convergence (retry). Raw: {response}",
                    diary_entry="",
                    completion_pct=pct_r,
                )
            corrective = ((parsed_retry or {}).get("corrective_instruction") or "").strip()
            if not corrective:
                raise RuntimeError(
                    f"Convergence retry returned no corrective instruction. Raw: {response[:200]}"
                )

        self._corrections_used += 1
        return DiaryCheckResult(
            action="command",
            new_instruction=corrective,
            reasoning=f"Convergence diagnosis: {parsed.get('diagnosis', 'unknown')}. Raw: {response}",
            diary_entry="",
            completion_pct=pct,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_frames_by_time(self) -> List[Path]:
        """Select frames closest to each ``check_interval_s`` boundary.

        Boundaries are placed at t=0, t=interval, t=2*interval, ... from the
        first timestamp. For each boundary, the frame with the smallest
        absolute time difference is chosen. The result is deduplicated while
        preserving order.
        """
        if not self._frame_timestamps or self._check_interval_s is None:
            return list(self._frame_paths)

        t0 = self._frame_timestamps[0]
        t_last = self._frame_timestamps[-1]
        interval = self._check_interval_s

        # Build boundary times: t0, t0+interval, t0+2*interval, ...
        boundaries: List[float] = []
        b = t0
        while b <= t_last:
            boundaries.append(b)
            b += interval
        # Always include a boundary at or beyond the last timestamp
        if not boundaries or boundaries[-1] < t_last:
            boundaries.append(t_last)

        selected: List[Path] = []
        seen_indices: set = set()
        for boundary in boundaries:
            best_idx = 0
            best_diff = abs(self._frame_timestamps[0] - boundary)
            for i, ts in enumerate(self._frame_timestamps):
                diff = abs(ts - boundary)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            if best_idx not in seen_indices:
                seen_indices.add(best_idx)
                selected.append(self._frame_paths[best_idx])

        return selected

    def _timed_query_vlm(self, grid: Any, prompt: str, label: str, **kwargs) -> str:
        t0 = time.time()
        response = query_vlm(grid, prompt, llm=self._llm, **kwargs)
        rtt = time.time() - t0
        self._vlm_calls += 1
        self._vlm_rtts.append({
            "label": label,
            "step": self._step,
            "rtt_s": round(rtt, 3),
        })
        return response

    @staticmethod
    def _make_llm(model: str) -> BaseLLM:
        if model.startswith("gemini"):
            return LLMFactory.create("gemini", model=model)
        return LLMFactory.create("openai", model=model)

    def _format_displacement(self) -> str:
        d = self._last_displacement
        return (
            f"[x: {d[0] / 100:.2f} m, y: {d[1] / 100:.2f} m, "
            f"z: {d[2] / 100:.2f} m, yaw: {d[3]:.1f}°]"
        )

    def _is_stalled(self) -> bool:
        """Return True if completion has plateaued over the last stall_window checkpoints."""
        history = self._completion_history
        if len(history) < self._stall_window:
            return False
        recent = history[-self._stall_window:]
        if min(recent) >= self._stall_completion_floor:
            return False
        return max(recent) - min(recent) < self._stall_threshold

    def _save_frame(self, frame: Any) -> Path:
        if isinstance(frame, (str, Path)):
            return Path(frame)
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="diary_monitor_")
        path = Path(self._temp_dir) / f"frame_{len(self._frame_paths):06d}.png"
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise RuntimeError("PIL is required to save numpy frames")
        PILImage.fromarray(frame.astype("uint8")).save(str(path))
        return path

    def _run_checkpoint(self) -> DiaryCheckResult:
        step = self._step
        n = self._check_interval

        prev_path = self._frame_paths[step - n]
        curr_path = self._frame_paths[step - 1]

        # --- Local query: what changed ---
        grid_two = build_frame_grid([prev_path, curr_path])
        prompt_local = LOCAL_PROMPT_TEMPLATE.format(subgoal=self._subgoal)
        change_text = self._timed_query_vlm(
            grid_two, prompt_local, "local_checkpoint",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        disp_str = self._format_displacement()
        diary_entry = f"Steps {step - n}-{step} {disp_str}: {change_text}"
        self._diary.append(diary_entry)

        # --- Global query: assess progress ---
        sampled = sample_frames_every_n(self._frame_paths, n)
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
        if parsed is None:
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
            if parsed is None:
                self._parse_failures += 1
                raise RuntimeError(
                    f"Checkpoint {step} JSON parse failed after retry. Raw: {response_global[:200]}"
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

    def _run_checkpoint_async(self) -> DiaryCheckResult:
        """Run a checkpoint from the background thread (time-based mode).

        Snapshots frame data under the lock, then releases the lock before
        making VLM calls.
        """
        with self._lock:
            if len(self._frame_paths) < 2:
                return DiaryCheckResult(
                    action="continue",
                    new_instruction="",
                    reasoning="Not enough frames for async checkpoint.",
                    diary_entry="",
                    completion_pct=self._last_completion_pct,
                )
            prev_path = self._frame_paths[-2]
            curr_path = self._frame_paths[-1]
            step = self._step
            displacement = list(self._last_displacement)

        # --- Local query: what changed ---
        grid_two = build_frame_grid([prev_path, curr_path])
        prompt_local = LOCAL_PROMPT_TEMPLATE.format(subgoal=self._subgoal)
        change_text = self._timed_query_vlm(
            grid_two, prompt_local, "local_checkpoint_async",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        # Use displacement snapshot for the diary entry
        d = displacement
        disp_str = (
            f"[x: {d[0] / 100:.2f} m, y: {d[1] / 100:.2f} m, "
            f"z: {d[2] / 100:.2f} m, yaw: {d[3]:.1f}\u00b0]"
        )
        diary_entry = f"Steps ~{step} {disp_str}: {change_text}"
        self._diary.append(diary_entry)

        # --- Global query: assess progress ---
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
            grid_global, prompt_global, "global_checkpoint_async",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )

        parsed = self._parse_json_response(response_global)
        if parsed is None:
            self._parse_failures += 1
            logger.warning(
                "Async checkpoint ~%d JSON parse failed, retrying. Raw: %s",
                step, response_global[:200],
            )
            response_global = self._timed_query_vlm(
                grid_global, prompt_global, "global_checkpoint_async_retry",
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            parsed = self._parse_json_response(response_global)
            if parsed is None:
                self._parse_failures += 1
                raise RuntimeError(
                    f"Async checkpoint ~{step} JSON parse failed after retry. "
                    f"Raw: {response_global[:200]}"
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

    def _run_convergence_async(
        self, frame_path: Path, displacement: List[float],
    ) -> DiaryCheckResult:
        """Run a convergence check from the background thread."""
        if self._corrections_used >= self._max_corrections:
            logger.warning(
                "Max corrections (%d) exhausted. Asking for help.",
                self._max_corrections,
            )
            return DiaryCheckResult(
                action="ask_help",
                new_instruction="",
                reasoning=f"Max corrections ({self._max_corrections}) exhausted.",
                diary_entry="",
                completion_pct=self._last_completion_pct,
            )

        self._last_displacement = list(displacement)
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
            grid, prompt, "convergence_async",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )

        self._save_convergence_artifact(response, prompt, grid)

        parsed = self._parse_json_response(response)
        if parsed is None:
            self._parse_failures += 1
            logger.warning(
                "Async convergence JSON parse failed (attempt 1), retrying. Raw: %s",
                response[:200],
            )
            response = self._timed_query_vlm(
                grid, prompt, "convergence_async_retry",
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            self._save_convergence_artifact(response, prompt, grid)
            parsed = self._parse_json_response(response)
            if parsed is None:
                self._parse_failures += 1
                raise RuntimeError(
                    f"Async convergence JSON parse failed after retry. Raw: {response[:200]}"
                )

        pct = float(parsed.get("completion_percentage", self._last_completion_pct))
        pct = max(0.0, min(1.0, pct))
        self._last_completion_pct = pct
        self._peak_completion = max(self._peak_completion, pct)

        if parsed.get("complete", False) or parsed.get("diagnosis") == "complete":
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning=f"Subgoal complete on async convergence. Raw: {response}",
                diary_entry="",
                completion_pct=pct,
            )

        corrective = parsed.get("corrective_instruction") or ""
        if not corrective:
            logger.warning(
                "Async convergence response missing corrective_instruction, retrying."
            )
            response = self._timed_query_vlm(
                grid, prompt, "convergence_async_instruction_retry",
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            self._save_convergence_artifact(response, prompt, grid)
            parsed_retry = self._parse_json_response(response)
            if parsed_retry is not None and (
                parsed_retry.get("complete", False) or parsed_retry.get("diagnosis") == "complete"
            ):
                pct_r = float(parsed_retry.get("completion_percentage", pct))
                pct_r = max(0.0, min(1.0, pct_r))
                self._last_completion_pct = pct_r
                self._peak_completion = max(self._peak_completion, pct_r)
                return DiaryCheckResult(
                    action="stop",
                    new_instruction="",
                    reasoning=f"Subgoal complete on async convergence (retry). Raw: {response}",
                    diary_entry="",
                    completion_pct=pct_r,
                )
            corrective = ((parsed_retry or {}).get("corrective_instruction") or "").strip()
            if not corrective:
                raise RuntimeError(
                    f"Async convergence retry returned no corrective instruction. "
                    f"Raw: {response[:200]}"
                )

        self._corrections_used += 1
        return DiaryCheckResult(
            action="command",
            new_instruction=corrective,
            reasoning=f"Async convergence diagnosis: {parsed.get('diagnosis', 'unknown')}. Raw: {response}",
            diary_entry="",
            completion_pct=pct,
        )

    def _monitor_loop(self) -> None:
        """Background thread main loop for time-based mode."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=0.05):
                break

            # Check for convergence request (higher priority)
            with self._lock:
                conv_req = self._convergence_request
                self._convergence_request = None

            if conv_req is not None:
                frame_path, displacement = conv_req
                try:
                    result = self._run_convergence_async(frame_path, displacement)
                except Exception:
                    logger.exception("Error in async convergence check")
                    result = DiaryCheckResult(
                        action="stop",
                        new_instruction="",
                        reasoning="async_convergence_error",
                        diary_entry="",
                        completion_pct=self._last_completion_pct,
                    )
                with self._lock:
                    self._pending_result = result
                continue

            # Check if checkpoint interval has elapsed
            now = time.time()
            if now - self._last_checkpoint_time >= (self._check_interval_s or 0):
                try:
                    result = self._run_checkpoint_async()
                except Exception:
                    logger.exception("Error in async checkpoint")
                    result = DiaryCheckResult(
                        action="force_converge",
                        new_instruction="",
                        reasoning="async_checkpoint_error",
                        diary_entry="",
                        completion_pct=self._last_completion_pct,
                    )
                with self._lock:
                    self._pending_result = result
                self._last_checkpoint_time = now

    def _parse_global_response(
        self, response: str, diary_entry: str,
        parsed: Optional[dict] = None,
    ) -> DiaryCheckResult:
        if parsed is None:
            parsed = self._parse_json_response(response)
            if parsed is None:
                raise RuntimeError(
                    f"Global response JSON parse failed. Raw: {response[:200]}"
                )
        pct = float(parsed.get("completion_percentage", self._last_completion_pct))
        pct = max(0.0, min(1.0, pct))

        if parsed.get("complete", False):
            return DiaryCheckResult(
                action="force_converge",
                new_instruction="",
                reasoning=f"Checkpoint thinks complete — verifying at convergence. Raw: {response}",
                diary_entry=diary_entry,
                completion_pct=pct,
            )

        if parsed.get("should_stop", False):
            return DiaryCheckResult(
                action="force_converge",
                new_instruction="",
                reasoning=f"Stop requested for correction. Raw: {response}",
                diary_entry=diary_entry,
                completion_pct=pct,
            )

        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning=f"On track ({pct:.0%}). Raw: {response}",
            diary_entry=diary_entry,
            completion_pct=pct,
        )

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict]:
        """Parse JSON from LLM response. Returns None on failure."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        logger.warning("Could not parse JSON from LLM response: %s", text[:200])
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

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

    def __del__(self) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Artifact saving
    # ------------------------------------------------------------------

    def _save_checkpoint_artifact(
        self,
        step: int,
        grid_local: Any,
        grid_global: Any,
        prompt_local: str,
        response_local: str,
        prompt_global: str,
        response_global: str,
    ) -> None:
        if self._artifacts_dir is None:
            return
        cp_dir = self._artifacts_dir / f"checkpoint_{step:04d}"
        cp_dir.mkdir(parents=True, exist_ok=True)
        grid_local.save(cp_dir / "grid_local.png")
        grid_global.save(cp_dir / "grid_global.png")
        (cp_dir / "prompt_local.txt").write_text(prompt_local)
        (cp_dir / "response_local.txt").write_text(response_local)
        (cp_dir / "prompt_global.txt").write_text(prompt_global)
        (cp_dir / "response_global.txt").write_text(response_global)
        diary_blob = "\n".join(self._diary)
        (cp_dir / "diary.txt").write_text(diary_blob)

    def _save_convergence_artifact(
        self, response: str, prompt: str, grid: Optional[Any] = None
    ) -> None:
        if self._artifacts_dir is None:
            return
        conv_dir = self._artifacts_dir / f"convergence_{self._corrections_used:03d}"
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "prompt.txt").write_text(f"{GENERAL_SYSTEM_PROMPT}\n\n{prompt}")
        (conv_dir / "response.txt").write_text(response)
        diary_blob = "\n".join(self._diary)
        (conv_dir / "diary.txt").write_text(diary_blob)
        if grid is not None:
            grid.save(conv_dir / "grid_convergence.png")

