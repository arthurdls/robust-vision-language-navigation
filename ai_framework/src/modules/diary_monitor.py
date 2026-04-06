"""
Live diary-based subgoal completion monitor with supervisor capabilities.

Every N steps during normal execution: builds a local 2-frame grid
(what changed?) and a global sampled grid (is the subgoal complete?
is the drone on-track?), tracks estimated completion percentage and
displacement, and returns stop / continue / override / command.

On convergence (drone stops): evaluates whether the subgoal is truly
complete or whether corrective commands should be issued to OpenVLA.

Key behaviours:
  - Completion percentage is reported as-is from the LLM (no clamping).
    A separate high-water mark tracks the peak value for analysis.
  - The global prompt asks the LLM for a ``should_override`` flag (rather
    than separate overshot / off-track flags).  When true and a
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

from .utils.base_llm_providers import BaseLLM, LLMFactory
from .utils.vision_utils import build_frame_grid, query_vlm, sample_frames_every_n

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiaryCheckResult:
    action: str           # "continue", "stop", "override", or "command"
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
- VISUAL SEARCH ("turn until you see X"): target is clearly visible.
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
  null only if complete."""



# ---------------------------------------------------------------------------
# LiveDiaryMonitor
# ---------------------------------------------------------------------------

class LiveDiaryMonitor:
    """Real-time diary-based subgoal completion monitor with supervisor capabilities.

    Designed to monitor a SINGLE subgoal (one predicate from the LTL planner).
    Operates in two modes:

    1. PASSIVE MONITORING (during normal execution):
       Every N steps: builds local 2-frame grid, builds global sampled grid
       (capped at 9 frames for a 3x3 layout), maintains running diary with
       displacement and completion percentage, and returns stop/continue/override.

    2. SUPERVISOR MODE (when drone converges / stops prematurely):
       When the control loop detects convergence but the subgoal is not complete,
       the monitor takes over and issues corrective commands directly to OpenVLA
       until the subgoal is achieved, a max correction budget is exhausted, or
       the monitor determines the subgoal was overshot and issues a reversal.
    """

    MAX_GLOBAL_FRAMES = 9

    def __init__(
        self,
        subgoal: str,
        check_interval: int,
        model: str = "gpt-4o",
        artifacts_dir: Optional[Path] = None,
        max_corrections: int = 15,
    ):
        self._subgoal = subgoal
        self._check_interval = check_interval
        self._model = model
        self._artifacts_dir = artifacts_dir
        self._max_corrections = max_corrections

        self._llm: BaseLLM = self._make_llm(model)
        self._frame_paths: List[Path] = []
        self._diary: List[str] = []
        self._step = 0
        self._corrections_used = 0
        self._parse_failures = 0
        self._vlm_calls = 0
        self._last_completion_pct: float = 0.0
        self._high_water_mark: float = 0.0
        self._last_displacement: List[float] = [0.0, 0.0, 0.0, 0.0]
        self._temp_dir: Optional[str] = None

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
    def high_water_mark(self) -> float:
        return self._high_water_mark

    @property
    def parse_failures(self) -> int:
        return self._parse_failures

    @property
    def vlm_calls(self) -> int:
        return self._vlm_calls

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
        self._frame_paths.append(path)
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
        instruction if not.
        """
        path = self._save_frame(latest_frame)
        if not self._frame_paths or self._frame_paths[-1] != path:
            self._frame_paths.append(path)

        if displacement is not None:
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

        sampled = sample_frames_every_n(self._frame_paths, self._check_interval)
        if not sampled or sampled[-1] != path:
            sampled.append(path)
        sampled = sampled[-self.MAX_GLOBAL_FRAMES:]

        grid = build_frame_grid(sampled)
        response = query_vlm(
            grid, prompt, llm=self._llm, system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        self._vlm_calls += 1

        self._save_convergence_artifact(response, prompt, grid)

        parsed = self._parse_json_response(response)
        if not parsed:
            self._parse_failures += 1
            logger.warning(
                "Convergence JSON parse failed (attempt 1), retrying. Raw: %s",
                response[:200],
            )
            response = query_vlm(
                grid, prompt, llm=self._llm, system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            self._vlm_calls += 1
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
            response = query_vlm(
                grid, prompt, llm=self._llm, system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            self._vlm_calls += 1
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        change_text = query_vlm(
            grid_two, prompt_local, llm=self._llm,
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        self._vlm_calls += 1
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

        response_global = query_vlm(
            grid_global, prompt_global, llm=self._llm,
            system_prompt=GENERAL_SYSTEM_PROMPT,
        )
        self._vlm_calls += 1

        parsed = self._parse_json_response(response_global)
        if not parsed:
            self._parse_failures += 1
            logger.warning(
                "Checkpoint %d JSON parse failed, retrying. Raw: %s",
                step, response_global[:200],
            )
            response_global = query_vlm(
                grid_global, prompt_global, llm=self._llm,
                system_prompt=GENERAL_SYSTEM_PROMPT,
            )
            self._vlm_calls += 1
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

    def _parse_global_response(
        self, response: str, diary_entry: str,
        parsed: Optional[dict] = None,
    ) -> DiaryCheckResult:
        if parsed is None:
            parsed = self._parse_json_response(response)
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
    def _parse_json_response(response: str) -> dict:
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
        return {}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove temporary frame directory if it was created."""
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

