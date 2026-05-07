"""
Live goal adherence subgoal completion monitor with supervisor capabilities.

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

Peak-dropoff override: tracks the per-subgoal peak completion estimate
``self._peak_completion``. If the peak has reached
``PEAK_DROPOFF_PEAK_FLOOR`` (0.9) at any point and the most recent
completion estimate has dropped at least ``PEAK_DROPOFF_DROP_THRESHOLD``
(0.25) below that peak, the monitor returns ``action="stop"`` and the
runner advances to the next subgoal, regardless of what the VLM
returned. Rationale: a peak >= 0.9 means the goal was at some point
near-complete, so the goal has probably been achieved; a >= 0.25 retreat
means the agent has clearly moved away, and continuing to issue
corrective instructions in that situation rarely recovers the goal.
The override fires in every place stall detection runs (sync/async
periodic checkpoints) plus the three convergence paths (sync image-grid,
async image-grid, text-only). It is a no-op for C1 and C3 because
neither has a monitor.

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

Corrective-action history (visible to the convergence VLM):
  The monitor maintains a per-subgoal log of every corrective it has
  issued during convergence checks (``self._correction_history``,
  exposed as ``correction_history``). Each entry is
  ``{step, diagnosis, completion_pct, instruction}``. Two delivery
  forms surface this log to the next convergence prompt, depending on
  whether the active prompt template displays the diary:

    * Diary-using templates (default DIARY_CONVERGENCE_PROMPT used by
      C0 / C2; TEXT_ONLY_CONVERGENCE_PROMPT used by C6): every
      issued corrective also drops a
      ``[CONVERGENCE @ step N]: corrective issued
      (<diagnosis>, <pct>% complete) -- "<instruction>"`` marker into
      ``self._diary``, so the convergence VLM sees past correctives
      interleaved with frame observations and can localise each one
      in the temporal record. The diary preface in those templates
      explains the marker convention and tells the VLM to switch
      axes if completion has not improved since the most recent
      marker. No separate corrections block is rendered.
    * Non-diary templates (GRID_ONLY / SINGLE_FRAME used by C5 / C4):
      ``_format_correction_history_block()`` renders the same log as
      an oldest-first ``corrections_block`` placeholder which those
      templates substitute. When the log is empty the block emits a
      "first convergence check" sentinel so the VLM does not
      hallucinate prior-correction reasoning on cycle 0.

  Both forms are written for every monitor-based condition; the
  marker/block split only affects which surface the convergence VLM
  sees the data through. The persisted ``diary`` (in
  ``diary_summary.json``) always carries the markers regardless of
  condition, so offline analysis can reconstruct the full temporal
  record. ``correction_history`` is also persisted directly. C1 and
  C3 have no monitor and no convergence cycle, so neither form
  applies. The history is implicitly reset per subgoal because
  ``GoalAdherenceMonitor`` is constructed fresh per subgoal in
  ``run_subgoal()``.

Correction-awareness gap:
  Periodic checkpoints (on_frame / _run_checkpoint) continue to run during
  corrective instruction execution, but they always evaluate against the
  *subgoal* (e.g. "fly to the red building"), not the active corrective
  micro-command (e.g. "move forward 2 meters"). The monitor has no concept
  of "correction mode": it cannot tell whether a correction is working,
  overshooting, or making things worse at the correction level. The only
  correction-specific evaluation happens at the *next* convergence, when
  on_convergence runs the full diagnostic prompt again.

  Temporal bounds: in sync mode (frame-based), up to check_interval - 1
  steps of blind execution after a correction before the next checkpoint.
  In async mode (time-based), up to check_interval_s seconds. Convergence
  detection is also suppressed for check_interval steps (sync) or
  check_interval_s seconds (async) after a correction to let the new
  instruction take effect before the small-motion detector kicks in.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ..config import (
    DEFAULT_MAX_CORRECTIONS,
    DEFAULT_STALL_COMPLETION_FLOOR,
    DEFAULT_STALL_THRESHOLD,
    DEFAULT_STALL_WINDOW,
    DEFAULT_VLM_MODEL,
    PEAK_DROPOFF_DROP_THRESHOLD,
    PEAK_DROPOFF_PEAK_FLOOR,
)
from .prompts import (
    DIARY_SYSTEM_PROMPT as GENERAL_SYSTEM_PROMPT,
    DIARY_LOCAL_PROMPT as LOCAL_PROMPT_TEMPLATE,
    DIARY_GLOBAL_PROMPT as GLOBAL_PROMPT_TEMPLATE,
    DIARY_CONVERGENCE_PROMPT as CONVERGENCE_PROMPT_TEMPLATE,
    TEXT_ONLY_GLOBAL_SYSTEM_PROMPT,
    TEXT_ONLY_GLOBAL_PROMPT as TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE,
    TEXT_ONLY_CONVERGENCE_PROMPT as TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE,
)
from .utils.llm_providers import BaseLLM, LLMFactory
from .utils.vision import build_frame_grid, query_vlm, sample_frames_every_n

logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, data: str) -> None:
    """Write text to ``path`` atomically.

    A reader observing ``path`` will see either the previous contents or the
    new contents, never a partial write. Implemented via write-to-tempfile +
    os.replace, which is atomic on POSIX (and reasonably atomic on Windows).
    Required for the pipelined monitor: the dashboard polls at 4 Hz while
    workers may rewrite ``diary.txt`` and similar files concurrently.
    """
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        tmp.write_text(data)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
DiaryAction = Literal["continue", "stop", "override", "command", "ask_help", "force_converge"]

# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiaryCheckResult:
    action: DiaryAction
    new_instruction: str  # populated when action is "override" or "command"
    reasoning: str
    diary_entry: str      # latest diary entry (what changed)
    completion_pct: float = 0.0  # latest estimated completion percentage
    # Optional short header shown to the operator when action="ask_help"
    # (e.g. "CONVERGENCE PARSE FAILURE", "MAX CORRECTIONS REACHED"). The
    # caller in interface.py uses it to label the help prompt; falls back
    # to "MAX CORRECTIONS REACHED" if empty for back-compat.
    ask_help_header: str = ""


@dataclass(frozen=True)
class _CheckpointSnapshot:
    """Immutable inputs a pipelined checkpoint worker operates on.

    Captured under the monitor lock at dispatch time so each in-flight worker
    has a stable view of frame data and the diary, regardless of what later
    dispatches mutate. Frozen so workers cannot accidentally write through
    the snapshot back to monitor state.
    """
    step: int
    displacement: List[float]
    frame_paths: List[Path]
    frame_timestamps: List[float]
    diary_at_dispatch: List[str]


@dataclass
class _LocalStageResult:
    step: int
    grid_local: Any  # PIL.Image or None when skip_local
    prompt_local: str
    response_local: str  # change_text from local VLM
    diary_entry: str  # what gets appended to _diary on commit; empty when skip_local


@dataclass
class _GlobalStageResult:
    step: int
    grid_global: Any  # PIL.Image or None when text_llm
    prompt_global: str
    response_global: str
    parsed: dict  # parsed JSON, with parse_failure_fallback if both retries failed
    system_prompt_global: str  # the actual system prompt used (varies by mode)

# ---------------------------------------------------------------------------
# GoalAdherenceMonitor
# ---------------------------------------------------------------------------

class GoalAdherenceMonitor:
    """Real-time goal adherence subgoal completion monitor with supervisor capabilities.

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
        model: str = DEFAULT_VLM_MODEL,
        artifacts_dir: Optional[Path] = None,
        max_corrections: int = DEFAULT_MAX_CORRECTIONS,
        check_interval_s: Optional[float] = None,
        global_grid_spacing_s: Optional[float] = None,
        local_grid_spacing_s: Optional[float] = None,
        stall_window: int = DEFAULT_STALL_WINDOW,
        stall_threshold: float = DEFAULT_STALL_THRESHOLD,
        stall_completion_floor: float = DEFAULT_STALL_COMPLETION_FLOOR,
        global_backend: Literal["vlm_grid", "text_llm"] = "vlm_grid",
        global_model: Optional[str] = None,
        single_frame_mode: bool = False,
        skip_local: bool = False,
    ):
        self._subgoal = subgoal
        self._check_interval = check_interval
        self._model = model
        self._artifacts_dir = artifacts_dir
        self._max_corrections = max_corrections

        # Single-frame ablation: skip local 2-frame VLM call, send only the
        # current frame to global/convergence VLM, do not maintain a diary.
        # Used by Condition 4 to test the value of temporal context.
        self._single_frame_mode: bool = single_frame_mode

        # Skip the local 2-frame VLM call without affecting the global grid.
        # Used by Condition 5 (grid-only) to avoid generating diary entries
        # that are never included in the global/convergence assessment,
        # saving ~2,300 tokens per checkpoint.
        self._skip_local: bool = skip_local or single_frame_mode

        # Text-only global backend configuration
        self._global_backend: Literal["vlm_grid", "text_llm"] = global_backend
        self._global_llm: Optional[BaseLLM] = None
        if global_backend == "text_llm":
            self._global_llm = self._make_llm(global_model or model)

        # Time-based mode configuration
        self._time_based: bool = check_interval_s is not None
        self._check_interval_s: Optional[float] = check_interval_s
        # Spacing (in seconds) between samples in the global VLM grid.
        # None falls back to the diary-check interval, so the
        # historical default (sample once per checkpoint) is preserved.
        # Setting it independently lets the global grid cover a wider
        # or narrower time window than the checkpoint cadence -- useful
        # when you want frequent monitor checks (small
        # check_interval_s) but a wider visual context window (larger
        # global_grid_spacing_s).
        self._global_grid_spacing_s: Optional[float] = (
            global_grid_spacing_s
            if global_grid_spacing_s is not None
            else check_interval_s
        )
        # Spacing (in seconds) between the prev and curr frames in the
        # 2-frame local "what changed" VLM grid (time-mode only). None
        # cascades down: local_grid_spacing_s -> global_grid_spacing_s
        # -> check_interval_s. Without this the time-mode local prompt
        # always compared the literal last two captured frames, which
        # are typically ~100 ms apart at 10 Hz capture -- too small a
        # delta for the VLM to articulate motion.
        self._local_grid_spacing_s: Optional[float] = (
            local_grid_spacing_s
            if local_grid_spacing_s is not None
            else self._global_grid_spacing_s
        )

        self._llm: BaseLLM = self._make_llm(model)
        self._frame_paths: List[Path] = []
        self._frame_timestamps: List[float] = []
        self._diary: List[str] = []
        # Running log of corrective actions issued during convergence checks
        # for this subgoal. Each entry is {step, diagnosis, completion_pct,
        # instruction}. Forwarded into subsequent convergence prompts (see
        # _format_correction_history_block) so the convergence VLM can avoid
        # reissuing a correction that already failed -- the analogue of the
        # diary, but for issued commands rather than observed frame changes.
        # Reset implicitly per subgoal because GoalAdherenceMonitor is
        # constructed fresh per subgoal in run_subgoal().
        self._correction_history: List[Dict[str, Any]] = []
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

        self._last_should_stop_reasoning: Optional[str] = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending_result: Optional[DiaryCheckResult] = None
        self._convergence_request: Optional[Tuple[Path, List[float]]] = None
        # True from request_convergence() until the matching convergence
        # result is committed. While set, any in-flight checkpoint that
        # finishes later must NOT overwrite _pending_result -- otherwise
        # _convergence_loop's poll picks up a stale checkpoint result and
        # the runner mistakes a stray force_converge for the convergence
        # verdict.
        self._awaiting_convergence: bool = False
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
    def correction_history(self) -> List[Dict[str, Any]]:
        """Copy of the corrective-action log for this subgoal (oldest first).

        Each entry has {step, diagnosis, completion_pct, instruction}. See
        _correction_history in __init__ for the rationale.
        """
        return [dict(entry) for entry in self._correction_history]

    @property
    def max_corrections(self) -> int:
        return self._max_corrections

    @property
    def corrections_exhausted(self) -> bool:
        return self._corrections_used >= self._max_corrections

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

    def reset_grace_state(self) -> None:
        """Reset the state that triggers operator help prompts.

        Called by interface.py when the operator dismisses an ask_help
        prompt with "continue with current instruction": without this,
        the very next checkpoint would see the same flat completion
        history and immediately re-stall, and the very next convergence
        would see _corrections_used >= max and immediately re-prompt with
        MAX CORRECTIONS REACHED. Clearing both gives the operator a fresh
        budget to actually let the run continue.
        """
        with self._lock:
            self._completion_history = []
            self._corrections_used = 0

    def request_convergence(
        self, frame_path: Union[Path, str], displacement: List[float],
    ) -> None:
        """Queue a convergence check for the background thread.

        Also discards any pending checkpoint result. Without this,
        _convergence_loop's tight poll could pick up a stale checkpoint
        force_converge that landed between the caller's last poll and now,
        and treat it as the convergence verdict; the subgoal would then
        terminate on a stray force_converge with no new_instruction.
        """
        with self._lock:
            self._convergence_request = (Path(frame_path), list(displacement))
            self._pending_result = None
            self._awaiting_convergence = True

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
        stop_reason = self._consume_stop_reasoning()

        if self._global_backend == "text_llm":
            return self._run_text_only_convergence(diary_blob, disp_str, stop_reason)

        prompt = self._format_convergence_prompt(diary_blob, disp_str, stop_reason)

        if self._single_frame_mode:
            # Single-frame ablation: convergence VLM sees only the current
            # frame, mirroring the checkpoint behaviour.
            grid = build_frame_grid([path])
        else:
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
                logger.error(
                    "Convergence JSON parse failed after retry, treating as "
                    "stopped_short with no corrective instruction. Raw: %s",
                    response[:200],
                )
                return DiaryCheckResult(
                    action="ask_help",
                    new_instruction="",
                    reasoning=(
                        "convergence_parse_failure: VLM JSON unparseable after "
                        f"retry. Raw: {response[:200]}"
                    ),
                    diary_entry="",
                    completion_pct=self._last_completion_pct,
                    ask_help_header="CONVERGENCE PARSE FAILURE",
                )

        pct = self._coerce_pct(
            parsed.get("completion_percentage"), self._last_completion_pct,
        )
        pct = max(0.0, min(1.0, pct))
        self._last_completion_pct = pct
        self._peak_completion = max(self._peak_completion, pct)

        corrective = (parsed.get("corrective_instruction") or "").strip()
        parsed_for_reasoning = parsed

        if self._coerce_bool(parsed.get("complete")):
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning=self._compose_convergence_reasoning(
                    parsed, "convergence: complete",
                ),
                diary_entry="",
                completion_pct=pct,
            )

        # Peak-dropoff override (sync convergence, image-grid backend).
        # See _peak_dropoff_override.
        override = self._peak_dropoff_override(pct)
        if override is not None:
            return override

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
            corrective_retry = ((parsed_retry or {}).get("corrective_instruction") or "").strip()
            if parsed_retry is not None and self._coerce_bool(parsed_retry.get("complete")):
                pct_r = self._coerce_pct(parsed_retry.get("completion_percentage"), pct)
                pct_r = max(0.0, min(1.0, pct_r))
                self._last_completion_pct = pct_r
                self._peak_completion = max(self._peak_completion, pct_r)
                return DiaryCheckResult(
                    action="stop",
                    new_instruction="",
                    reasoning=self._compose_convergence_reasoning(
                        parsed_retry, "convergence: complete (on retry)",
                    ),
                    diary_entry="",
                    completion_pct=pct_r,
                )
            corrective = corrective_retry
            if corrective and parsed_retry is not None:
                parsed_for_reasoning = parsed_retry
            if not corrective:
                logger.warning(
                    "Convergence retry returned no corrective instruction; "
                    "asking operator for help. Raw: %s",
                    response[:200],
                )
                return DiaryCheckResult(
                    action="ask_help",
                    new_instruction="",
                    reasoning=(
                        f"convergence_no_corrective: VLM did not return a "
                        f"corrective instruction after retry. Raw: {response[:200]}"
                    ),
                    diary_entry="",
                    completion_pct=pct,
                    ask_help_header="CONVERGENCE GAVE NO CORRECTIVE",
                )

        self._corrections_used += 1
        self._record_correction(
            instruction=corrective,
            diagnosis=parsed_for_reasoning.get("diagnosis"),
            pct=pct,
        )
        return DiaryCheckResult(
            action="command",
            new_instruction=corrective,
            reasoning=self._compose_convergence_reasoning(
                parsed_for_reasoning, "convergence",
            ),
            diary_entry="",
            completion_pct=pct,
        )

    def _run_text_only_convergence(
        self, diary_blob: str, disp_str: str, stop_reason: Optional[str] = None,
    ) -> DiaryCheckResult:
        """Run text-only convergence check (no image grid)."""
        prompt = TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
            stop_reasoning_block=self._stop_reasoning_block(stop_reason),
            corrections_block=self._format_correction_history_block(),
        )

        messages = [
            {"role": "system", "content": TEXT_ONLY_GLOBAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        t0 = time.time()
        response = self._global_llm.make_request(messages, temperature=0.0)
        rtt = time.time() - t0
        self._vlm_calls += 1
        usage = self._global_llm.last_usage
        self._vlm_rtts.append({
            "label": "text_only_convergence",
            "step": self._step,
            "rtt_s": round(rtt, 3),
            "model": usage.get("model", self._model),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })
        self._save_convergence_artifact(
            response, prompt, grid=None,
            system_prompt=TEXT_ONLY_GLOBAL_SYSTEM_PROMPT,
        )

        parsed = self._parse_json_response(response)
        if parsed is None:
            self._parse_failures += 1
            logger.warning(
                "Text-only convergence JSON parse failed (attempt 1), retrying. Raw: %s",
                response[:200],
            )
            t0 = time.time()
            response = self._global_llm.make_request(messages, temperature=0.0)
            rtt = time.time() - t0
            self._vlm_calls += 1
            usage = self._global_llm.last_usage
            self._vlm_rtts.append({
                "label": "text_only_convergence_retry",
                "step": self._step,
                "rtt_s": round(rtt, 3),
                "model": usage.get("model", self._model),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            })
            self._save_convergence_artifact(
                response, prompt, grid=None,
                system_prompt=TEXT_ONLY_GLOBAL_SYSTEM_PROMPT,
            )
            parsed = self._parse_json_response(response)
            if parsed is None:
                self._parse_failures += 1
                logger.error(
                    "Text-only convergence parse failed after retry; "
                    "asking operator for help. Raw: %s", response[:200],
                )
                return DiaryCheckResult(
                    action="ask_help",
                    new_instruction="",
                    reasoning=(
                        "text_only_convergence_parse_failure: VLM JSON "
                        f"unparseable after retry. Raw: {response[:200]}"
                    ),
                    diary_entry="",
                    completion_pct=self._last_completion_pct,
                    ask_help_header="CONVERGENCE PARSE FAILURE",
                )

        pct = self._coerce_pct(
            parsed.get("completion_percentage"), self._last_completion_pct,
        )
        pct = max(0.0, min(1.0, pct))
        self._last_completion_pct = pct
        self._peak_completion = max(self._peak_completion, pct)

        corrective = (parsed.get("corrective_instruction") or "").strip()
        parsed_for_reasoning = parsed

        if self._coerce_bool(parsed.get("complete")):
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning=self._compose_convergence_reasoning(
                    parsed, "text-only convergence: complete",
                ),
                diary_entry="",
                completion_pct=pct,
            )

        # Peak-dropoff override (text-only convergence backend).
        # See _peak_dropoff_override.
        override = self._peak_dropoff_override(pct)
        if override is not None:
            return override

        if not corrective:
            t0 = time.time()
            response = self._global_llm.make_request(messages, temperature=0.0)
            rtt = time.time() - t0
            self._vlm_calls += 1
            usage = self._global_llm.last_usage
            self._vlm_rtts.append({
                "label": "text_only_convergence_instruction_retry",
                "step": self._step,
                "rtt_s": round(rtt, 3),
                "model": usage.get("model", self._model),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            })
            self._save_convergence_artifact(
                response, prompt, grid=None,
                system_prompt=TEXT_ONLY_GLOBAL_SYSTEM_PROMPT,
            )
            parsed_retry = self._parse_json_response(response)
            corrective_retry = ((parsed_retry or {}).get("corrective_instruction") or "").strip()
            if parsed_retry is not None and self._coerce_bool(parsed_retry.get("complete")):
                pct_r = self._coerce_pct(parsed_retry.get("completion_percentage"), pct)
                pct_r = max(0.0, min(1.0, pct_r))
                self._last_completion_pct = pct_r
                self._peak_completion = max(self._peak_completion, pct_r)
                return DiaryCheckResult(
                    action="stop",
                    new_instruction="",
                    reasoning=self._compose_convergence_reasoning(
                        parsed_retry, "text-only convergence: complete (on retry)",
                    ),
                    diary_entry="",
                    completion_pct=pct_r,
                )
            corrective = corrective_retry
            if corrective and parsed_retry is not None:
                parsed_for_reasoning = parsed_retry
            if not corrective:
                logger.warning(
                    "Text-only convergence retry: no corrective instruction; "
                    "asking operator for help. Raw: %s", response[:200],
                )
                return DiaryCheckResult(
                    action="ask_help",
                    new_instruction="",
                    reasoning=(
                        "text_only_convergence_no_corrective: VLM did not "
                        f"return a corrective instruction after retry. "
                        f"Raw: {response[:200]}"
                    ),
                    diary_entry="",
                    completion_pct=pct,
                    ask_help_header="CONVERGENCE GAVE NO CORRECTIVE",
                )

        self._corrections_used += 1
        self._record_correction(
            instruction=corrective,
            diagnosis=parsed_for_reasoning.get("diagnosis"),
            pct=pct,
        )
        return DiaryCheckResult(
            action="command",
            new_instruction=corrective,
            reasoning=self._compose_convergence_reasoning(
                parsed_for_reasoning, "text-only convergence",
            ),
            diary_entry="",
            completion_pct=pct,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_local_prev_path(
        self,
        frame_paths: List[Path],
        frame_timestamps: List[float],
    ) -> Optional[Path]:
        """Pick the prev-frame for the time-mode local 'what changed' grid.

        Looks `local_grid_spacing_s` seconds back from the most recent
        timestamp and returns the frame whose timestamp is closest to
        that target. If the spacing is None (or zero) or the history
        doesn't go back that far, falls back to the literal previous
        frame (legacy behavior).
        """
        if not frame_paths or not frame_timestamps:
            return None
        spacing = self._local_grid_spacing_s
        if spacing is None or spacing <= 0:
            return frame_paths[-2] if len(frame_paths) >= 2 else frame_paths[-1]
        target = frame_timestamps[-1] - spacing
        if target <= frame_timestamps[0]:
            return frame_paths[0]
        # Linear scan; checkpoint cadence keeps frame_paths small enough
        # that this is cheap and avoids bisect bookkeeping.
        best_idx = 0
        best_diff = abs(frame_timestamps[0] - target)
        for i, ts in enumerate(frame_timestamps):
            diff = abs(ts - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return frame_paths[best_idx]

    def _sample_frames_by_time(
        self,
        frame_paths: Optional[List[Path]] = None,
        frame_timestamps: Optional[List[float]] = None,
    ) -> List[Path]:
        """Select frames closest to each ``check_interval_s`` boundary.

        Boundaries are placed strictly at t=0, t=interval, t=2*interval, ...
        from the first timestamp -- they are FIXED across checkpoints.
        Earlier versions also appended a boundary at t_last (the most
        recent frame's timestamp) so the latest frame would always be
        in the grid; that broke consistency, since t_last moves every
        checkpoint and so the second-to-last grid cell jumped around
        instead of being a stable shift-by-one. With fixed boundaries
        every existing grid cell shows the same frame at every
        checkpoint, and a new cell is appended only when a new boundary
        is crossed (~once per checkpoint at the default 3 s cadence).

        For each boundary, the frame with the smallest absolute time
        difference is chosen. The result is deduplicated while
        preserving order.

        ``frame_paths`` and ``frame_timestamps`` may be pre-snapshotted
        copies taken under the lock by callers running on background
        threads.
        """
        paths = frame_paths if frame_paths is not None else self._frame_paths
        timestamps = frame_timestamps if frame_timestamps is not None else self._frame_timestamps

        interval = self._global_grid_spacing_s
        if not timestamps or interval is None:
            return list(paths)

        t0 = timestamps[0]
        t_last = timestamps[-1]

        # Build boundary times at FIXED multiples of interval. Do NOT
        # append t_last as a final boundary -- that would make the
        # rightmost grid cell move every checkpoint and defeat the
        # shift-by-one consistency property.
        boundaries: List[float] = []
        b = t0
        while b <= t_last:
            boundaries.append(b)
            b += interval

        selected: List[Path] = []
        seen_indices: set = set()
        for boundary in boundaries:
            best_idx = 0
            best_diff = abs(timestamps[0] - boundary)
            for i, ts in enumerate(timestamps):
                diff = abs(ts - boundary)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            if best_idx not in seen_indices:
                seen_indices.add(best_idx)
                selected.append(paths[best_idx])

        return selected

    def _timed_query_vlm(self, grid: Any, prompt: str, label: str, **kwargs) -> str:
        t0 = time.time()
        response = query_vlm(grid, prompt, llm=self._llm, **kwargs)
        rtt = time.time() - t0
        self._vlm_calls += 1
        usage = self._llm.last_usage
        self._vlm_rtts.append({
            "label": label,
            "step": self._step,
            "rtt_s": round(rtt, 3),
            "model": usage.get("model", self._model),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })
        return response

    @staticmethod
    def _make_llm(model: str) -> BaseLLM:
        return LLMFactory.create("openai", model=model)

    def _format_displacement(self) -> str:
        d = self._last_displacement
        return (
            f"[x: {d[0] / 100:.2f} m, y: {d[1] / 100:.2f} m, "
            f"z: {d[2] / 100:.2f} m, yaw: {d[3]:.1f}°]"
        )

    def _format_global_prompt(self, diary_blob: str, disp_str: str) -> str:
        return GLOBAL_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )

    def _format_global_prompt_with_diary(self, diary_blob: str, disp_str: str) -> str:
        """Variant of _format_global_prompt that takes an explicit diary blob.

        The pipelined async path needs to use a snapshot diary, not self._diary,
        because workers run on private snapshots and self._diary may have
        already advanced.
        """
        return GLOBAL_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )

    @staticmethod
    def _stop_reasoning_block(reason: Optional[str]) -> str:
        """Single-line context describing why the drone was stopped.

        Forwarded into convergence prompts so the convergence VLM does not have
        to re-derive why the drone was stopped (e.g., it can carry forward
        "altitude dropped below 10 m" instead of re-evaluating from scratch).
        Returns an empty string for natural drone-stops (no should_stop).
        """
        text = (reason or "").strip()
        if not text:
            return ""
        return f"The monitor stopped the drone because: {text}\n"

    def _consume_stop_reasoning(self) -> str:
        """Pop the most recent should_stop reasoning so it applies to one
        convergence cycle only and is not carried into the next one."""
        reason = self._last_should_stop_reasoning or ""
        self._last_should_stop_reasoning = None
        return reason

    def _format_correction_history_block(self) -> str:
        """Format the corrective-action history for the convergence prompt.

        Mirrors how the diary is formatted: an explicit header plus an
        oldest-first list. When empty, returns a sentinel that tells the
        VLM this is the first convergence check so the
        avoid-repeating-a-correction guidance does not apply (otherwise
        the model can second-guess a correct first-cycle verdict on
        nonexistent prior history). When non-empty, the block also
        carries the directive to switch axes on repeated failures, so
        the static prompt body stays neutral about whether corrections
        have happened yet.
        """
        if not self._correction_history:
            return (
                "Past corrective actions issued for this subgoal: "
                "(none yet -- this is the first convergence check for "
                "this subgoal, so reasoning about prior corrections does "
                "not apply.)\n"
            )
        lines = ["Past corrective actions issued for this subgoal (oldest first):"]
        for i, entry in enumerate(self._correction_history, start=1):
            diag = entry.get("diagnosis") or "unknown"
            pct = float(entry.get("completion_pct") or 0.0)
            instr = (entry.get("instruction") or "").strip()
            lines.append(
                f"  {i}. [{diag}, {pct * 100:.0f}% complete] \"{instr}\""
            )
        lines.append(
            "If completion has not improved since a recent correction, switch "
            "axes (e.g., try altitude or a different turn direction) instead of "
            "reissuing the same command."
        )
        return "\n".join(lines) + "\n"

    def _record_correction(
        self, instruction: str, diagnosis: Optional[str], pct: float,
    ) -> None:
        """Append a corrective-action entry. Called at each site that
        increments _corrections_used so the history and the counter stay
        in lockstep. Safe under the GIL; list.append is atomic and the
        only concurrent reader (correction_history property) copies.

        Also appends a marker line to the diary so the next convergence
        prompt sees WHERE in the observation timeline each corrective
        was applied. Without this, the convergence VLM gets the diary
        and the corrections list as two unrelated sequences and cannot
        tell which diary entries describe behaviour BEFORE vs AFTER a
        given correction. The marker uses a distinct ``[CONVERGENCE @
        step N]`` prefix so it is unambiguously metadata, not a frame
        observation.
        """
        diag = diagnosis or "unknown"
        instr = (instruction or "").strip()
        self._correction_history.append({
            "step": self._step,
            "diagnosis": diag,
            "completion_pct": float(pct),
            "instruction": instr,
        })
        self._diary.append(
            f"[CONVERGENCE @ step {self._step}]: corrective issued "
            f"({diag}, {pct * 100:.0f}% complete) -- \"{instr}\""
        )

    @staticmethod
    def _compose_convergence_reasoning(parsed: dict, label: str) -> str:
        """Build the operator-facing reasoning string for a convergence verdict.

        Prefers the VLM's own "reasoning" field (a plain-English explanation
        the prompt asks for) so the [CONVERGENCE] terminal print shows what
        the VLM actually saw, not just the diagnosis label. Raw response text
        is intentionally NOT embedded -- the full JSON lives in the saved
        convergence artifact files for forensic review.
        """
        diagnosis = str(parsed.get("diagnosis", "unknown"))
        vlm_reasoning = (parsed.get("reasoning") or "").strip()
        if vlm_reasoning:
            return f"{label} ({diagnosis}): {vlm_reasoning}"
        return f"{label} ({diagnosis}); no reasoning provided"

    def _format_convergence_prompt(
        self, diary_blob: str, disp_str: str, stop_reason: Optional[str] = None,
    ) -> str:
        return CONVERGENCE_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
            stop_reasoning_block=self._stop_reasoning_block(stop_reason),
            corrections_block=self._format_correction_history_block(),
        )

    def _is_stalled(self) -> bool:
        """Return True if completion has been flat over the last stall_window
        checkpoints and we're below the completion floor.

        Previously this also gated on `recent_peak >= peak_completion -
        threshold`, intending to suppress stall after a one-off VLM
        hallucination spike. In practice that gate also suppressed stall
        whenever completion regressed from any prior value -- including the
        common "drone stuck near 0 the entire run" case where a single
        slightly-higher early estimate kept the operator from ever being
        pulled in. We just look at the recent window now: flat below the
        floor = stalled. A real hallucination spike still delays detection
        only by stall_window checkpoints (until it falls off the recent
        window), which is acceptable.
        """
        history = self._completion_history
        if len(history) < self._stall_window:
            return False
        recent = history[-self._stall_window:]
        # Don't pull the operator if we're already nearly done.
        if min(recent) >= self._stall_completion_floor:
            return False
        # Flat = stalled. Any meaningful upward swing within the window
        # means progress was made, so don't fire.
        return (max(recent) - min(recent)) < self._stall_threshold

    def _peak_dropoff_override(self, pct: float) -> Optional[DiaryCheckResult]:
        """Return a stop-action result if the peak-dropoff rule fires, else None.

        Rule: if the highest completion estimate seen so far for this
        subgoal is at least ``PEAK_DROPOFF_PEAK_FLOOR`` (0.9), and the
        freshly observed completion estimate ``pct`` has dropped at
        least ``PEAK_DROPOFF_DROP_THRESHOLD`` (0.25) below that peak,
        treat the subgoal as achieved and return action="stop".

        Why: a peak >= 0.9 means the goal was at some point
        near-complete, so the goal has probably been achieved. A drop
        >= 0.25 means the agent has clearly moved away from the
        achieved state. Continuing to issue corrective instructions in
        that situation rarely recovers the goal and burns correction
        budget; advancing to the next subgoal is the better call.

        The helper appends a diary entry recording the override so the
        decision is visible in saved checkpoint and convergence
        artifacts. Callers are responsible for not invoking this helper
        when the VLM has already proposed an "stop" action -- in that
        case there is nothing to override and re-attributing the stop
        to peak-dropoff would corrupt the reasoning string.
        """
        if self._peak_completion < PEAK_DROPOFF_PEAK_FLOOR:
            return None
        if (self._peak_completion - pct) < PEAK_DROPOFF_DROP_THRESHOLD:
            return None
        diary_entry = (
            f"Peak dropoff override (step {self._step}): "
            f"peak={self._peak_completion:.2f}, current={pct:.2f}, "
            "treating subgoal as achieved"
        )
        self._diary.append(diary_entry)
        reasoning = (
            f"peak_dropoff_override: peak={self._peak_completion:.2f}, "
            f"current={pct:.2f}, treating subgoal as achieved "
            f"(peak >= {PEAK_DROPOFF_PEAK_FLOOR}, "
            f"drop >= {PEAK_DROPOFF_DROP_THRESHOLD})"
        )
        return DiaryCheckResult(
            action="stop",
            new_instruction="",
            reasoning=reasoning,
            diary_entry=diary_entry,
            completion_pct=pct,
        )

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

    def _snapshot_for_checkpoint(self) -> Optional["_CheckpointSnapshot"]:
        """Capture the inputs a pipelined worker needs, under the lock.

        Returns None when there are not yet enough frames to form a 2-frame
        local grid. The caller (dispatcher) skips submission in that case.
        """
        with self._lock:
            if len(self._frame_paths) < 2:
                return None
            return _CheckpointSnapshot(
                step=self._step,
                displacement=list(self._last_displacement),
                frame_paths=list(self._frame_paths),
                frame_timestamps=list(self._frame_timestamps),
                diary_at_dispatch=list(self._diary),
            )

    def _run_checkpoint(self) -> DiaryCheckResult:
        step = self._step
        n = self._check_interval

        prev_path = self._frame_paths[step - n]
        curr_path = self._frame_paths[step - 1]

        # --- Local query: what changed (uses VLM with 2-frame comparison) ---
        # Skipped when _skip_local is set: Condition 4 (single_frame_mode)
        # and Condition 5 (grid_only) both skip the local VLM call to avoid
        # generating diary entries that are never used in their
        # global/convergence assessments.
        if self._skip_local:
            change_text = ""
            grid_two = None
            prompt_local = ""
            disp_str = self._format_displacement()
            diary_entry = ""
        else:
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
        diary_blob = "\n".join(self._diary)

        if self._global_backend == "text_llm":
            response_global, prompt_global = self._run_text_only_global(diary_blob, disp_str, step)
            grid_global = None
        elif self._single_frame_mode:
            # Single-frame ablation: send ONLY the current frame, not a
            # 9-frame grid. Use the SINGLE_FRAME_GLOBAL_PROMPT templates
            # patched in by subgoal_runner._patch_monitor_prompts.
            grid_global = build_frame_grid([curr_path])
            prompt_global = self._format_global_prompt(diary_blob, disp_str)
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
                    logger.error(
                        "Checkpoint %d JSON parse failed after retry; "
                        "treating as continue with prior completion. Raw: %s",
                        step, response_global[:200],
                    )
                    parsed = {
                        "complete": False,
                        "completion_percentage": self._last_completion_pct,
                        "should_stop": False,
                        "reasoning": "parse_failure_fallback",
                    }
        else:
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
            prompt_global = self._format_global_prompt(diary_blob, disp_str)

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
                    logger.error(
                        "Checkpoint %d JSON parse failed after retry; "
                        "treating as continue with prior completion. Raw: %s",
                        step, response_global[:200],
                    )
                    parsed = {
                        "complete": False,
                        "completion_percentage": self._last_completion_pct,
                        "should_stop": False,
                        "reasoning": "parse_failure_fallback",
                    }

        if self._global_backend == "text_llm":
            parsed = self._parse_json_response(response_global)
            if parsed is None:
                self._parse_failures += 1
                logger.warning(
                    "Text-only checkpoint %d JSON parse failed, retrying. Raw: %s",
                    step, response_global[:200],
                )
                response_global, prompt_global = self._run_text_only_global(
                    diary_blob, disp_str, step, label_suffix="_retry",
                )
                parsed = self._parse_json_response(response_global)
                if parsed is None:
                    self._parse_failures += 1
                    logger.error(
                        "Text-only checkpoint %d JSON parse failed after "
                        "retry; treating as continue with prior completion. "
                        "Raw: %s", step, response_global[:200],
                    )
                    parsed = {
                        "complete": False,
                        "completion_percentage": self._last_completion_pct,
                        "should_stop": False,
                        "reasoning": "parse_failure_fallback",
                    }

        # Save checkpoint artifacts in EVERY mode, including text-only
        # (grid_global is None there). The saver gates image writes on grid
        # presence, so passing None is safe and ensures M5 has a decision
        # trail for C6 too.
        sysp_global = (
            TEXT_ONLY_GLOBAL_SYSTEM_PROMPT
            if self._global_backend == "text_llm"
            else GENERAL_SYSTEM_PROMPT
        )
        self._save_checkpoint_artifact(
            step, grid_two, grid_global,
            prompt_local, change_text,
            prompt_global, response_global,
            system_prompt_global=sysp_global,
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

        # Peak-dropoff override: if completion previously hit
        # PEAK_DROPOFF_PEAK_FLOOR and has dropped >= PEAK_DROPOFF_DROP_THRESHOLD,
        # the goal was probably achieved and the agent has drifted; advance
        # to the next subgoal instead of issuing more corrections. Gated on
        # action == "continue" to mirror stall detection: only intervene when
        # the VLM is not already taking action. See _peak_dropoff_override.
        if result.action == "continue":
            override = self._peak_dropoff_override(result.completion_pct)
            if override is not None:
                return override

        return result

    def _run_text_only_global(self, diary_blob: str, disp_str: str, step: int, label_suffix: str = "") -> Tuple[str, str]:
        """Run text-only global assessment (no image grid).

        Returns (response, prompt) so callers can persist the actual
        prompt that was sent (artifact logging for C6).
        """
        prompt = TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )

        messages = [
            {"role": "system", "content": TEXT_ONLY_GLOBAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        label = f"text_only_global{label_suffix}"
        t0 = time.time()
        response = self._global_llm.make_request(messages, temperature=0.0)
        rtt = time.time() - t0
        self._vlm_calls += 1
        usage = self._global_llm.last_usage
        self._vlm_rtts.append({
            "label": label,
            "step": step,
            "rtt_s": round(rtt, 3),
            "model": usage.get("model", self._model),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })
        return response, prompt

    def _run_local_stage(self, snap: "_CheckpointSnapshot") -> "_LocalStageResult":
        """Run the local 2-frame VLM call for a pipelined checkpoint.

        Pure with respect to monitor state: reads the snapshot only, returns
        a result. The publisher commits the diary entry to ``_diary``.
        """
        curr_path = snap.frame_paths[-1]
        prev_path = self._pick_local_prev_path(snap.frame_paths, snap.frame_timestamps)
        if prev_path is None:
            prev_path = curr_path

        d = snap.displacement
        disp_str = (
            f"[x: {d[0] / 100:.2f} m, y: {d[1] / 100:.2f} m, "
            f"z: {d[2] / 100:.2f} m, yaw: {d[3]:.1f}°]"
        )
        if self._skip_local:
            return _LocalStageResult(
                step=snap.step,
                grid_local=None,
                prompt_local="",
                response_local="",
                diary_entry="",
            )
        grid_two = build_frame_grid([prev_path, curr_path])
        prompt_local = LOCAL_PROMPT_TEMPLATE.format(subgoal=self._subgoal)
        change_text = self._timed_query_vlm(
            grid_two, prompt_local, "local_checkpoint_async",
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        diary_entry = f"Steps ~{snap.step} {disp_str}: {change_text}"
        return _LocalStageResult(
            step=snap.step,
            grid_local=grid_two,
            prompt_local=prompt_local,
            response_local=change_text,
            diary_entry=diary_entry,
        )

    def _run_global_stage(
        self, snap: "_CheckpointSnapshot", local: "_LocalStageResult",
    ) -> "_GlobalStageResult":
        """Run the global progress-assessment VLM call for a pipelined checkpoint.

        Sees the diary as it appeared at dispatch time, with this checkpoint's
        own local entry appended (best-effort visibility: earlier still-in-flight
        checkpoints will not yet appear). Includes the parse-failure retry path
        and falls back to a synthetic ``parsed`` dict if both attempts fail, so
        the publisher always receives a usable result.
        """
        d = snap.displacement
        disp_str = (
            f"[x: {d[0] / 100:.2f} m, y: {d[1] / 100:.2f} m, "
            f"z: {d[2] / 100:.2f} m, yaw: {d[3]:.1f}°]"
        )
        diary_lines = list(snap.diary_at_dispatch)
        if local.diary_entry:
            diary_lines.append(local.diary_entry)
        diary_blob = "\n".join(diary_lines)

        if self._global_backend == "text_llm":
            response_global, prompt_global = self._run_text_only_global(
                diary_blob, disp_str, snap.step,
            )
            grid_global = None
            sysp_global = TEXT_ONLY_GLOBAL_SYSTEM_PROMPT
        elif self._single_frame_mode:
            grid_global = build_frame_grid([snap.frame_paths[-1]])
            prompt_global = self._format_global_prompt_with_diary(diary_blob, disp_str)
            response_global = self._timed_query_vlm(
                grid_global, prompt_global, "global_checkpoint_async",
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            sysp_global = GENERAL_SYSTEM_PROMPT
        else:
            sampled = self._sample_frames_by_time(snap.frame_paths, snap.frame_timestamps)
            sampled = sampled[-self.MAX_GLOBAL_FRAMES:] if sampled else [snap.frame_paths[-1]]
            grid_global = build_frame_grid(sampled)
            prompt_global = self._format_global_prompt_with_diary(diary_blob, disp_str)
            response_global = self._timed_query_vlm(
                grid_global, prompt_global, "global_checkpoint_async",
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            sysp_global = GENERAL_SYSTEM_PROMPT

        parsed = self._parse_json_response(response_global)
        if parsed is None:
            self._parse_failures += 1
            logger.warning(
                "Async checkpoint ~%d JSON parse failed, retrying. Raw: %s",
                snap.step, response_global[:200],
            )
            if self._global_backend == "text_llm":
                response_global, prompt_global = self._run_text_only_global(
                    diary_blob, disp_str, snap.step, label_suffix="_async_retry",
                )
            else:
                response_global = self._timed_query_vlm(
                    grid_global, prompt_global, "global_checkpoint_async_retry",
                    system_prompt=GENERAL_SYSTEM_PROMPT,
                )
            parsed = self._parse_json_response(response_global)
            if parsed is None:
                self._parse_failures += 1
                logger.error(
                    "Async checkpoint ~%d JSON parse failed after retry; "
                    "treating as continue with prior completion. Raw: %s",
                    snap.step, response_global[:200],
                )
                parsed = {
                    "complete": False,
                    "completion_percentage": self._last_completion_pct,
                    "should_stop": False,
                    "reasoning": "parse_failure_fallback",
                }
        return _GlobalStageResult(
            step=snap.step,
            grid_global=grid_global,
            prompt_global=prompt_global,
            response_global=response_global,
            parsed=parsed,
            system_prompt_global=sysp_global,
        )

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
            curr_path = self._frame_paths[-1]
            step = self._step
            displacement = list(self._last_displacement)
            frame_paths_snap = list(self._frame_paths)
            frame_timestamps_snap = list(self._frame_timestamps)
        # Pick prev OUTSIDE the lock to avoid holding it while doing the
        # linear scan. The snapshots above are immutable copies.
        prev_path = self._pick_local_prev_path(
            frame_paths_snap, frame_timestamps_snap,
        )
        if prev_path is None:
            prev_path = curr_path

        # --- Local query: what changed (skipped when _skip_local is set) ---
        d = displacement
        disp_str = (
            f"[x: {d[0] / 100:.2f} m, y: {d[1] / 100:.2f} m, "
            f"z: {d[2] / 100:.2f} m, yaw: {d[3]:.1f}\u00b0]"
        )
        if self._skip_local:
            grid_two = None
            prompt_local = ""
            change_text = ""
            diary_entry = ""
        else:
            grid_two = build_frame_grid([prev_path, curr_path])
            prompt_local = LOCAL_PROMPT_TEMPLATE.format(subgoal=self._subgoal)
            change_text = self._timed_query_vlm(
                grid_two, prompt_local, "local_checkpoint_async",
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            diary_entry = f"Steps ~{step} {disp_str}: {change_text}"
            self._diary.append(diary_entry)

        # --- Global query: assess progress ---
        diary_blob = "\n".join(self._diary)

        if self._global_backend == "text_llm":
            response_global, prompt_global = self._run_text_only_global(diary_blob, disp_str, step)
            grid_global = None
        elif self._single_frame_mode:
            grid_global = build_frame_grid([curr_path])
            prompt_global = self._format_global_prompt(diary_blob, disp_str)
            response_global = self._timed_query_vlm(
                grid_global, prompt_global, "global_checkpoint_async",
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )
        else:
            sampled = self._sample_frames_by_time(frame_paths_snap, frame_timestamps_snap)
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
            prompt_global = self._format_global_prompt(diary_blob, disp_str)
            response_global = self._timed_query_vlm(
                grid_global, prompt_global, "global_checkpoint_async",
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )

        if self._global_backend == "text_llm":
            parsed = self._parse_json_response(response_global)
            if parsed is None:
                self._parse_failures += 1
                logger.warning(
                    "Async text-only checkpoint ~%d JSON parse failed, retrying. Raw: %s",
                    step, response_global[:200],
                )
                response_global, prompt_global = self._run_text_only_global(
                    diary_blob, disp_str, step, label_suffix="_async_retry",
                )
                parsed = self._parse_json_response(response_global)
                if parsed is None:
                    self._parse_failures += 1
                    logger.error(
                        "Async text-only checkpoint ~%d JSON parse failed after "
                        "retry; treating as continue with prior completion. Raw: %s",
                        step, response_global[:200],
                    )
                    parsed = {
                        "complete": False,
                        "completion_percentage": self._last_completion_pct,
                        "should_stop": False,
                        "reasoning": "parse_failure_fallback",
                    }
        else:
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
                    logger.error(
                        "Async checkpoint ~%d JSON parse failed after retry; "
                        "treating as continue with prior completion. Raw: %s",
                        step, response_global[:200],
                    )
                    parsed = {
                        "complete": False,
                        "completion_percentage": self._last_completion_pct,
                        "should_stop": False,
                        "reasoning": "parse_failure_fallback",
                    }

        # Save artifacts in every mode, including text-only (grid_global=None).
        sysp_global = (
            TEXT_ONLY_GLOBAL_SYSTEM_PROMPT
            if self._global_backend == "text_llm"
            else GENERAL_SYSTEM_PROMPT
        )
        self._save_checkpoint_artifact(
            step, grid_two, grid_global,
            prompt_local, change_text,
            prompt_global, response_global,
            system_prompt_global=sysp_global,
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

        # Peak-dropoff override (async checkpoint). See _peak_dropoff_override.
        if result.action == "continue":
            override = self._peak_dropoff_override(result.completion_pct)
            if override is not None:
                return override

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
        stop_reason = self._consume_stop_reasoning()

        if self._global_backend == "text_llm":
            return self._run_text_only_convergence(diary_blob, disp_str, stop_reason)

        prompt = self._format_convergence_prompt(diary_blob, disp_str, stop_reason)

        if self._single_frame_mode:
            grid = build_frame_grid([frame_path])
        else:
            with self._lock:
                frame_paths_snap = list(self._frame_paths)
                frame_timestamps_snap = list(self._frame_timestamps)
            sampled = self._sample_frames_by_time(frame_paths_snap, frame_timestamps_snap)
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
                logger.error(
                    "Async convergence JSON parse failed after retry; "
                    "asking operator for help. Raw: %s", response[:200],
                )
                return DiaryCheckResult(
                    action="ask_help",
                    new_instruction="",
                    reasoning=(
                        "async_convergence_parse_failure: VLM JSON "
                        f"unparseable after retry. Raw: {response[:200]}"
                    ),
                    diary_entry="",
                    completion_pct=self._last_completion_pct,
                    ask_help_header="CONVERGENCE PARSE FAILURE",
                )

        pct = self._coerce_pct(
            parsed.get("completion_percentage"), self._last_completion_pct,
        )
        pct = max(0.0, min(1.0, pct))
        self._last_completion_pct = pct
        self._peak_completion = max(self._peak_completion, pct)

        corrective = (parsed.get("corrective_instruction") or "").strip()
        parsed_for_reasoning = parsed

        if self._coerce_bool(parsed.get("complete")):
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning=self._compose_convergence_reasoning(
                    parsed, "async convergence: complete",
                ),
                diary_entry="",
                completion_pct=pct,
            )

        # Peak-dropoff override (async convergence, image-grid backend).
        # See _peak_dropoff_override.
        override = self._peak_dropoff_override(pct)
        if override is not None:
            return override

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
            corrective_retry = ((parsed_retry or {}).get("corrective_instruction") or "").strip()
            if parsed_retry is not None and self._coerce_bool(parsed_retry.get("complete")):
                pct_r = self._coerce_pct(parsed_retry.get("completion_percentage"), pct)
                pct_r = max(0.0, min(1.0, pct_r))
                self._last_completion_pct = pct_r
                self._peak_completion = max(self._peak_completion, pct_r)
                return DiaryCheckResult(
                    action="stop",
                    new_instruction="",
                    reasoning=self._compose_convergence_reasoning(
                        parsed_retry, "async convergence: complete (on retry)",
                    ),
                    diary_entry="",
                    completion_pct=pct_r,
                )
            corrective = corrective_retry
            if corrective and parsed_retry is not None:
                parsed_for_reasoning = parsed_retry
            if not corrective:
                logger.warning(
                    "Async convergence retry returned no corrective instruction; "
                    "asking operator for help. Raw: %s", response[:200],
                )
                return DiaryCheckResult(
                    action="ask_help",
                    new_instruction="",
                    reasoning=(
                        "async_convergence_no_corrective: VLM did not return "
                        f"a corrective instruction after retry. Raw: {response[:200]}"
                    ),
                    diary_entry="",
                    completion_pct=pct,
                    ask_help_header="CONVERGENCE GAVE NO CORRECTIVE",
                )

        self._corrections_used += 1
        self._record_correction(
            instruction=corrective,
            diagnosis=parsed_for_reasoning.get("diagnosis"),
            pct=pct,
        )
        return DiaryCheckResult(
            action="command",
            new_instruction=corrective,
            reasoning=self._compose_convergence_reasoning(
                parsed_for_reasoning, "async convergence",
            ),
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
                except Exception as exc:
                    logger.exception("Error in async convergence check")
                    result = DiaryCheckResult(
                        action="ask_help",
                        new_instruction="",
                        reasoning=f"async_convergence_error: {exc}",
                        diary_entry="",
                        ask_help_header="CONVERGENCE VLM ERROR",
                        completion_pct=self._last_completion_pct,
                    )
                with self._lock:
                    self._pending_result = result
                    self._awaiting_convergence = False
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
                    # If a convergence was requested while this checkpoint
                    # was in flight, drop the result -- the operator-visible
                    # next event must be the convergence verdict, not a
                    # stale force_converge.
                    if not self._awaiting_convergence:
                        self._pending_result = result
                self._last_checkpoint_time = now

    def _parse_global_response(
        self, response: str, diary_entry: str,
        parsed: Optional[dict] = None,
    ) -> DiaryCheckResult:
        if parsed is None:
            parsed = self._parse_json_response(response)
            if parsed is None:
                self._parse_failures += 1
                logger.warning(
                    "Global response parse failed inside _parse_global_response; "
                    "falling back to continue. Raw: %s", response[:200],
                )
                parsed = {
                    "complete": False,
                    "completion_percentage": self._last_completion_pct,
                    "should_stop": False,
                    "reasoning": "parse_failure_fallback",
                }
        pct = self._coerce_pct(
            parsed.get("completion_percentage"), self._last_completion_pct,
        )
        pct = max(0.0, min(1.0, pct))
        model_reasoning = str(parsed.get("reasoning", "") or "").strip()

        if self._coerce_bool(parsed.get("complete")):
            self._last_should_stop_reasoning = (
                model_reasoning or "checkpoint thinks subgoal complete; verifying at convergence"
            )
            return DiaryCheckResult(
                action="force_converge",
                new_instruction="",
                reasoning=f"Checkpoint thinks complete, verifying at convergence. Reasoning: {model_reasoning} Raw: {response}",
                diary_entry=diary_entry,
                completion_pct=pct,
            )

        if self._coerce_bool(parsed.get("should_stop")):
            self._last_should_stop_reasoning = (
                model_reasoning or "monitor flagged off-track or hazard"
            )
            return DiaryCheckResult(
                action="force_converge",
                new_instruction="",
                reasoning=f"Stop requested for correction. Reasoning: {model_reasoning} Raw: {response}",
                diary_entry=diary_entry,
                completion_pct=pct,
            )

        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning=f"On track ({pct:.0%}). Reasoning: {model_reasoning} Raw: {response}",
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

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """Strict bool coercion. Native bools pass through. Numbers coerce
        through Python truthiness. Strings are accepted as True only via an
        explicit allow-list ("true", "yes", "1"); everything else is False.

        This guards against LLMs returning the string "false" (truthy under
        bool() in Python) as a value for boolean fields like "complete".
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return False

    @staticmethod
    def _coerce_pct(value: Any, fallback: float) -> float:
        """Coerce a JSON value to a float in [0, 1], with a fallback.

        Returns the fallback when value is None, missing, or non-numeric.
        Guards against TypeError when the LLM returns null for the
        completion_percentage field.
        """
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop background thread and remove temporary frame directory."""
        if not hasattr(self, "_stop_event"):
            return
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
        system_prompt_global: Optional[str] = None,
    ) -> None:
        if self._artifacts_dir is None:
            return
        cp_dir = self._artifacts_dir / f"checkpoint_{step:04d}"
        cp_dir.mkdir(parents=True, exist_ok=True)
        if grid_local is not None:
            grid_local.save(cp_dir / "grid_local.png")
        if grid_global is not None:
            grid_global.save(cp_dir / "grid_global.png")
        if prompt_local:
            (cp_dir / "prompt_local.txt").write_text(prompt_local)
        if response_local:
            (cp_dir / "response_local.txt").write_text(response_local)
        # Record the actual system prompt for the global call (matters for
        # text-only / single-frame / grid-only ablations where it differs
        # from the default DIARY_SYSTEM_PROMPT). Falls back to the patched
        # module-level GENERAL_SYSTEM_PROMPT if the caller doesn't pass it.
        sysp = system_prompt_global if system_prompt_global is not None else GENERAL_SYSTEM_PROMPT
        (cp_dir / "system_prompt_global.txt").write_text(sysp)
        (cp_dir / "prompt_global.txt").write_text(prompt_global)
        (cp_dir / "response_global.txt").write_text(response_global)
        diary_blob = "\n".join(self._diary)
        (cp_dir / "diary.txt").write_text(diary_blob)

    def _save_convergence_artifact(
        self,
        response: str,
        prompt: str,
        grid: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if self._artifacts_dir is None:
            return
        conv_dir = self._artifacts_dir / f"convergence_{self._corrections_used:03d}"
        conv_dir.mkdir(parents=True, exist_ok=True)
        # Multiple VLM calls can occur within one convergence event (initial
        # query, parse-failure retry, missing-instruction retry). Append a
        # zero-padded suffix so retries don't clobber each other.
        retry_idx = 0
        while (conv_dir / f"prompt_{retry_idx:02d}.txt").exists():
            retry_idx += 1
        # Use the actual system prompt if provided (e.g. text-only path uses
        # TEXT_ONLY_GLOBAL_SYSTEM_PROMPT, not GENERAL_SYSTEM_PROMPT). Falling
        # back to GENERAL_SYSTEM_PROMPT preserves legacy behavior for callers
        # that don't pass it explicitly.
        sysp = system_prompt if system_prompt is not None else GENERAL_SYSTEM_PROMPT
        (conv_dir / f"prompt_{retry_idx:02d}.txt").write_text(
            f"{sysp}\n\n{prompt}"
        )
        (conv_dir / f"response_{retry_idx:02d}.txt").write_text(response)
        diary_blob = "\n".join(self._diary)
        (conv_dir / "diary.txt").write_text(diary_blob)
        if grid is not None:
            grid.save(conv_dir / f"grid_convergence_{retry_idx:02d}.png")

