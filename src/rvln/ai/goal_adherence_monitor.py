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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ..config import DEFAULT_MAX_CORRECTIONS, DEFAULT_VLM_MODEL
from .prompts import (
    DIARY_SYSTEM_PROMPT as GENERAL_SYSTEM_PROMPT,
    DIARY_LOCAL_PROMPT as LOCAL_PROMPT_TEMPLATE,
    DIARY_GLOBAL_PROMPT as GLOBAL_PROMPT_TEMPLATE,
    DIARY_GLOBAL_PROMPT_WITH_CONSTRAINTS as GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS,
    DIARY_CONVERGENCE_PROMPT as CONVERGENCE_PROMPT_TEMPLATE,
    DIARY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS as CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS,
    TEXT_ONLY_GLOBAL_SYSTEM_PROMPT,
    TEXT_ONLY_GLOBAL_PROMPT as TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE,
    TEXT_ONLY_GLOBAL_PROMPT_WITH_CONSTRAINTS as TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS,
    TEXT_ONLY_CONVERGENCE_PROMPT as TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE,
    TEXT_ONLY_CONVERGENCE_PROMPT_WITH_CONSTRAINTS as TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS,
)
from .utils.llm_providers import BaseLLM, LLMFactory
from .utils.vision import build_frame_grid, query_vlm, sample_frames_every_n

logger = logging.getLogger(__name__)


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
    constraint_violated: bool = False

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
        stall_window: int = 3,
        stall_threshold: float = 0.05,
        stall_completion_floor: float = 0.8,
        constraints: Optional[List[Any]] = None,
        negative_constraints: Optional[List[str]] = None,
        global_backend: Literal["vlm_grid", "text_llm"] = "vlm_grid",
        global_model: Optional[str] = None,
    ):
        self._subgoal = subgoal
        self._check_interval = check_interval
        self._model = model
        self._artifacts_dir = artifacts_dir
        self._max_corrections = max_corrections
        self._constraints: List[Any] = list(
            constraints or negative_constraints or []
        )

        # Text-only global backend configuration
        self._global_backend: Literal["vlm_grid", "text_llm"] = global_backend
        self._global_llm: Optional[BaseLLM] = None
        if global_backend == "text_llm":
            self._global_llm = self._make_llm(global_model or model)

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

        if self._global_backend == "text_llm":
            return self._run_text_only_convergence(diary_blob, disp_str)

        prompt = self._format_convergence_prompt(diary_blob, disp_str)

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

    def _run_text_only_convergence(self, diary_blob: str, disp_str: str) -> DiaryCheckResult:
        """Run text-only convergence check (no image grid)."""
        if self._constraints:
            prompt = TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS.format(
                subgoal=self._subgoal,
                diary=diary_blob,
                prev_completion_pct=self._last_completion_pct,
                displacement=disp_str,
                constraints_block=self._constraints_block(),
            )
        else:
            prompt = TEXT_ONLY_CONVERGENCE_PROMPT_TEMPLATE.format(
                subgoal=self._subgoal,
                diary=diary_blob,
                prev_completion_pct=self._last_completion_pct,
                displacement=disp_str,
                constraints_block="",
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
            parsed = self._parse_json_response(response)
            if parsed is None:
                self._parse_failures += 1
                raise RuntimeError(
                    f"Text-only convergence parse failed after retry. Raw: {response[:200]}"
                )

        pct = float(parsed.get("completion_percentage", self._last_completion_pct))
        pct = max(0.0, min(1.0, pct))
        self._last_completion_pct = pct
        self._peak_completion = max(self._peak_completion, pct)

        if parsed.get("complete", False) or parsed.get("diagnosis") == "complete":
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning=f"Text-only convergence: complete. Raw: {response}",
                diary_entry="",
                completion_pct=pct,
            )

        corrective = parsed.get("corrective_instruction") or ""
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
                    reasoning=f"Text-only convergence: complete (retry). Raw: {response}",
                    diary_entry="",
                    completion_pct=pct_r,
                )
            corrective = ((parsed_retry or {}).get("corrective_instruction") or "").strip()
            if not corrective:
                raise RuntimeError(
                    f"Text-only convergence retry: no corrective instruction. Raw: {response[:200]}"
                )

        self._corrections_used += 1
        return DiaryCheckResult(
            action="command",
            new_instruction=corrective,
            reasoning=f"Text-only convergence: {parsed.get('diagnosis', 'unknown')}. Raw: {response}",
            diary_entry="",
            completion_pct=pct,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_frames_by_time(
        self,
        frame_paths: Optional[List[Path]] = None,
        frame_timestamps: Optional[List[float]] = None,
    ) -> List[Path]:
        """Select frames closest to each ``check_interval_s`` boundary.

        Boundaries are placed at t=0, t=interval, t=2*interval, ... from the
        first timestamp. For each boundary, the frame with the smallest
        absolute time difference is chosen. The result is deduplicated while
        preserving order.

        ``frame_paths`` and ``frame_timestamps`` may be pre-snapshotted copies
        taken under the lock by callers running on background threads.
        """
        paths = frame_paths if frame_paths is not None else self._frame_paths
        timestamps = frame_timestamps if frame_timestamps is not None else self._frame_timestamps

        if not timestamps or self._check_interval_s is None:
            return list(paths)

        t0 = timestamps[0]
        t_last = timestamps[-1]
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

    def _constraints_block(self) -> str:
        """Return a prompt section listing active constraints, or empty string."""
        if not self._constraints:
            return ""
        lines = ["Active constraints (must be maintained throughout):"]
        for c in self._constraints:
            if hasattr(c, "polarity"):
                label = "AVOID" if c.polarity == "negative" else "MAINTAIN"
                lines.append(f"  - {label}: {c.description}")
            else:
                lines.append(f"  - {c}")
        lines.append("")
        return "\n".join(lines)

    def _format_global_prompt(self, diary_blob: str, disp_str: str) -> str:
        if self._constraints:
            return GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS.format(
                subgoal=self._subgoal,
                diary=diary_blob,
                prev_completion_pct=self._last_completion_pct,
                displacement=disp_str,
                constraints_block=self._constraints_block(),
            )
        return GLOBAL_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )

    def _format_convergence_prompt(self, diary_blob: str, disp_str: str) -> str:
        if self._constraints:
            return CONVERGENCE_PROMPT_TEMPLATE_CONSTRAINTS.format(
                subgoal=self._subgoal,
                diary=diary_blob,
                prev_completion_pct=self._last_completion_pct,
                displacement=disp_str,
                constraints_block=self._constraints_block(),
            )
        return CONVERGENCE_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
            prev_completion_pct=self._last_completion_pct,
            displacement=disp_str,
        )

    def _is_stalled(self) -> bool:
        """Return True if peak completion has not increased over the last stall_window checkpoints.

        Uses peak_completion as the baseline rather than raw history, which is
        more robust to non-monotonic VLM estimates (e.g., a hallucinated high
        value followed by truthful lower values would not trigger false stalls).
        """
        history = self._completion_history
        if len(history) < self._stall_window:
            return False
        recent = history[-self._stall_window:]
        if min(recent) >= self._stall_completion_floor:
            return False
        recent_peak = max(recent)
        if recent_peak >= self._peak_completion - self._stall_threshold:
            return max(recent) - min(recent) < self._stall_threshold
        return False

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

        # --- Local query: what changed (always uses VLM with images) ---
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
            response_global = self._run_text_only_global(diary_blob, disp_str, step)
            grid_global = None
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
                    raise RuntimeError(
                        f"Checkpoint {step} JSON parse failed after retry. Raw: {response_global[:200]}"
                    )

        if self._global_backend == "text_llm":
            parsed = self._parse_json_response(response_global)
            if parsed is None:
                self._parse_failures += 1
                logger.warning(
                    "Text-only checkpoint %d JSON parse failed, retrying. Raw: %s",
                    step, response_global[:200],
                )
                response_global = self._run_text_only_global(diary_blob, disp_str, step, label_suffix="_retry")
                parsed = self._parse_json_response(response_global)
                if parsed is None:
                    self._parse_failures += 1
                    raise RuntimeError(
                        f"Text-only checkpoint {step} JSON parse failed after retry. Raw: {response_global[:200]}"
                    )

        if grid_global is not None:
            self._save_checkpoint_artifact(
                step, grid_two, grid_global,
                prompt_local, change_text,
                self._format_global_prompt(diary_blob, disp_str), response_global,
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

    def _run_text_only_global(self, diary_blob: str, disp_str: str, step: int, label_suffix: str = "") -> str:
        """Run text-only global assessment (no image grid)."""
        if self._constraints:
            prompt = TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE_CONSTRAINTS.format(
                subgoal=self._subgoal,
                diary=diary_blob,
                prev_completion_pct=self._last_completion_pct,
                displacement=disp_str,
                constraints_block=self._constraints_block(),
            )
        else:
            prompt = TEXT_ONLY_GLOBAL_PROMPT_TEMPLATE.format(
                subgoal=self._subgoal,
                diary=diary_blob,
                prev_completion_pct=self._last_completion_pct,
                displacement=disp_str,
                constraints_block="",
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
        return response

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
            frame_paths_snap = list(self._frame_paths)
            frame_timestamps_snap = list(self._frame_timestamps)

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
        diary_blob = "\n".join(self._diary)
        prompt_global = self._format_global_prompt(diary_blob, disp_str)

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
        prompt = self._format_convergence_prompt(diary_blob, disp_str)

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

        if self._constraints and parsed.get("constraint_violated", False):
            return DiaryCheckResult(
                action="force_converge",
                new_instruction="",
                reasoning=f"Constraint violated, stopping for correction. Raw: {response}",
                diary_entry=diary_entry,
                completion_pct=pct,
                constraint_violated=True,
            )

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

