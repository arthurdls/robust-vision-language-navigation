"""
Live diary-based subgoal completion monitor with supervisor capabilities.

Every N steps during normal execution: builds a local 2-frame grid
(what changed?) and a global sampled grid (is the subgoal complete?
is the drone on-track?), and returns stop / continue / override / command.

On convergence (drone stops): evaluates whether the subgoal is truly
complete or whether corrective commands should be issued to OpenVLA.
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# System prompts (general -- not per-task)
# ---------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = """\
You are a subgoal completion monitor and supervisor for an autonomous drone.
You evaluate whether a single navigation/visual subgoal has been achieved based
on the drone's first-person video frames and a running diary of observed changes.
When the drone stops prematurely or overshoots, you issue corrective commands.

General completion criteria:
- MOVEMENT subgoals ("move past X", "go between X and Y"): complete when visual
  evidence shows the drone has reached or passed the described spatial relationship
  relative to the landmark.
- APPROACH subgoals ("approach X", "get close to X"): complete when the target
  occupies a large portion of the frame and appears close.
- VISUAL SEARCH subgoals ("turn until you see X"): complete when the target object
  is clearly visible in the drone's field of view.
- ALTITUDE subgoals ("go above X"): complete when the scene perspective shows the
  drone is above the target (target visible below).
- TRAVERSAL subgoals ("move through X"): complete when the drone has passed through
  the described structure and it is no longer directly ahead.

Judge based on the progression of visual evidence across the diary, not a single frame.

EARLY STOPPING: If the diary shows the subgoal has JUST been achieved or is about to be
achieved (e.g., the target is now clearly visible, the drone has reached the landmark),
immediately signal completion to stop the drone. Do not wait for the drone to keep moving
-- stopping promptly prevents overshoot.

When issuing corrective commands:
- Use short, imperative drone instructions (e.g., "turn right", "move forward",
  "turn left slightly").
- If the drone stopped too early (subgoal not yet achieved), issue commands to
  continue toward the goal.
- If the drone overshot (went past the goal), issue reversal commands in natural
  language (e.g., "turn left slightly", "move backward") to undo the overshoot.
- You may issue multiple corrections in sequence until satisfied."""

LOCAL_PROMPT_TEMPLATE = """\
The subgoal is: {subgoal}

What changed between these two consecutive frames relative to this subgoal?
Answer in ONE short sentence with only the key facts that directly bear on the subgoal."""

GLOBAL_PROMPT_TEMPLATE = """\
Subgoal: {subgoal}

Diary of changes observed so far:
{diary}

Based on the diary and the grid of sampled frames (left to right, top to bottom,
in temporal order), respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "on_track": true/false,
  "overshot": true/false,
  "corrective_instruction": "..." or null
}}

- "complete": true if the subgoal is achieved.
- "on_track": true if the drone is making progress toward the subgoal.
- "overshot": true if the drone went past the goal.
- "corrective_instruction": if off-track or overshot, a short imperative drone
  command to fix it; null otherwise."""

CONVERGENCE_PROMPT_TEMPLATE = """\
Subgoal: {subgoal}

Diary of changes observed so far:
{diary}

The drone has stopped moving. Given the diary and the latest frame, is the
subgoal complete? If not, did the drone stop short or overshoot?

Respond with EXACTLY ONE JSON object (no markdown fences):

{{
  "complete": true/false,
  "diagnosis": "stopped_short" or "overshot" or "complete",
  "corrective_instruction": "..." or null
}}

- "complete": true if the subgoal is fully achieved.
- "diagnosis": "complete" if done, "stopped_short" if the drone needs to keep
  going, "overshot" if the drone went past the goal.
- "corrective_instruction": if not complete, a short imperative drone command
  to fix it; null if complete."""


# ---------------------------------------------------------------------------
# LiveDiaryMonitor
# ---------------------------------------------------------------------------

class LiveDiaryMonitor:
    """Real-time diary-based subgoal completion monitor with supervisor capabilities.

    Designed to monitor a SINGLE subgoal (one predicate from the LTL planner).
    Operates in two modes:

    1. PASSIVE MONITORING (during normal execution):
       Every N steps: builds local 2-frame grid, builds global sampled grid,
       maintains running diary, and returns stop/continue/override.

    2. SUPERVISOR MODE (when drone converges / stops prematurely):
       When the control loop detects convergence but the subgoal is not complete,
       the monitor takes over and issues corrective commands directly to OpenVLA
       until the subgoal is achieved, a max correction budget is exhausted, or
       the monitor determines the subgoal was overshot and issues a reversal.
    """

    def __init__(
        self,
        subgoal: str,
        check_interval: int,
        model: str = "gpt-4o",
        artifacts_dir: Optional[Path] = None,
        max_corrections: int = 10,
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

    def on_frame(self, frame_image_or_path: Union[np.ndarray, Path, str]) -> DiaryCheckResult:
        """Process one frame during normal execution.

        On non-checkpoint steps returns action="continue" with empty diary_entry.
        On checkpoint steps (every check_interval) runs two LLM queries and
        returns the assessed action.
        """
        path = self._save_frame(frame_image_or_path)
        self._frame_paths.append(path)
        self._step += 1

        if self._step % self._check_interval != 0 or self._step < self._check_interval:
            return DiaryCheckResult(
                action="continue",
                new_instruction="",
                reasoning="",
                diary_entry="",
            )

        return self._run_checkpoint()

    def on_convergence(self, latest_frame: Union[np.ndarray, Path, str]) -> DiaryCheckResult:
        """Called when the control loop detects the drone has stopped (convergence).

        Evaluates whether the subgoal is truly complete or if corrective commands
        are needed. Returns 'stop' if complete, 'command' with a corrective
        instruction if not.
        """
        path = self._save_frame(latest_frame)
        if not self._frame_paths or self._frame_paths[-1] != path:
            self._frame_paths.append(path)

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
            )

        diary_blob = "\n".join(self._diary) if self._diary else "(no diary entries yet)"
        prompt = CONVERGENCE_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
        )

        response = query_vlm(
            build_frame_grid([path]),
            f"{GENERAL_SYSTEM_PROMPT}\n\n{prompt}",
            llm=self._llm,
        )

        self._save_convergence_artifact(response, prompt)

        parsed = self._parse_json_response(response)

        if parsed.get("complete", False) or parsed.get("diagnosis") == "complete":
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning=f"Subgoal complete on convergence. Raw: {response}",
                diary_entry="",
            )

        corrective = parsed.get("corrective_instruction") or ""
        if not corrective:
            diagnosis = parsed.get("diagnosis", "stopped_short")
            if diagnosis == "overshot":
                corrective = "turn around slowly"
            else:
                corrective = "continue forward"

        self._corrections_used += 1
        return DiaryCheckResult(
            action="command",
            new_instruction=corrective,
            reasoning=f"Convergence diagnosis: {parsed.get('diagnosis', 'unknown')}. Raw: {response}",
            diary_entry="",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_llm(model: str) -> BaseLLM:
        if model.startswith("gemini"):
            return LLMFactory.create("gemini", model=model)
        return LLMFactory.create("openai", model=model)

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
            grid_two,
            f"{GENERAL_SYSTEM_PROMPT}\n\n{prompt_local}",
            llm=self._llm,
        )
        diary_entry = f"Steps {step - n}-{step}: {change_text}"
        self._diary.append(diary_entry)

        # --- Global query: assess progress ---
        sampled = sample_frames_every_n(self._frame_paths, n)
        if not sampled:
            return DiaryCheckResult(
                action="continue",
                new_instruction="",
                reasoning="No sampled frames for global grid.",
                diary_entry=diary_entry,
            )

        grid_global = build_frame_grid(sampled)
        diary_blob = "\n".join(self._diary)
        prompt_global = GLOBAL_PROMPT_TEMPLATE.format(
            subgoal=self._subgoal,
            diary=diary_blob,
        )

        response_global = query_vlm(
            grid_global,
            f"{GENERAL_SYSTEM_PROMPT}\n\n{prompt_global}",
            llm=self._llm,
        )

        self._save_checkpoint_artifact(
            step, grid_two, grid_global,
            prompt_local, change_text,
            prompt_global, response_global,
        )

        return self._parse_global_response(response_global, diary_entry)

    def _parse_global_response(self, response: str, diary_entry: str) -> DiaryCheckResult:
        parsed = self._parse_json_response(response)

        if parsed.get("complete", False):
            return DiaryCheckResult(
                action="stop",
                new_instruction="",
                reasoning=f"Subgoal complete. Raw: {response}",
                diary_entry=diary_entry,
            )

        if parsed.get("overshot", False):
            corrective = parsed.get("corrective_instruction") or "turn around slowly"
            return DiaryCheckResult(
                action="override",
                new_instruction=corrective,
                reasoning=f"Overshot detected. Raw: {response}",
                diary_entry=diary_entry,
            )

        if not parsed.get("on_track", True):
            corrective = parsed.get("corrective_instruction") or ""
            if corrective:
                return DiaryCheckResult(
                    action="override",
                    new_instruction=corrective,
                    reasoning=f"Off-track. Raw: {response}",
                    diary_entry=diary_entry,
                )

        return DiaryCheckResult(
            action="continue",
            new_instruction="",
            reasoning=f"On track. Raw: {response}",
            diary_entry=diary_entry,
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

    def _save_convergence_artifact(self, response: str, prompt: str) -> None:
        if self._artifacts_dir is None:
            return
        conv_dir = self._artifacts_dir / f"convergence_{self._corrections_used:03d}"
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "prompt.txt").write_text(f"{GENERAL_SYSTEM_PROMPT}\n\n{prompt}")
        (conv_dir / "response.txt").write_text(response)
        diary_blob = "\n".join(self._diary)
        (conv_dir / "diary.txt").write_text(diary_blob)
