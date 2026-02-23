"""
Goal adherence monitor: uses a VLM to verify from image history whether
the current subgoal or full goal is achieved, and whether to suggest retry.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from .utils.base_llm_providers import LLMFactory

logger = logging.getLogger(__name__)


@dataclass
class GoalMonitorResult:
    """Result of checking goal adherence from image(s)."""

    subgoal_achieved: bool
    progress_made: bool
    goal_achieved: bool
    suggest_retry: bool


class GoalAdherenceMonitor:
    """
    VLM-based monitor: given recent image(s), current subgoal, and optional full goal,
    returns whether the subgoal/full goal is achieved and whether to retry.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self._llm = LLMFactory.create(model=model, rate_limit_seconds=0.0)

    def check(
        self,
        image_history: Union[List[np.ndarray], "np.ndarray"],
        current_subgoal: str,
        full_goal: Optional[str] = None,
        model_claimed_done: bool = False,
    ) -> GoalMonitorResult:
        """
        Check goal adherence from image history.

        Args:
            image_history: List of RGB numpy images (H,W,3) or a single image.
            current_subgoal: The short-horizon subgoal text we are verifying.
            full_goal: Optional full natural-language goal (for progress/final check).
            model_claimed_done: True if the low-level model reported done for this subgoal.

        Returns:
            GoalMonitorResult with subgoal_achieved, progress_made, goal_achieved, suggest_retry.
        """
        if isinstance(image_history, np.ndarray):
            images = [image_history]
        else:
            images = list(image_history) if image_history else []

        if not images:
            logger.warning("Goal monitor called with no images; returning conservative result.")
            return GoalMonitorResult(
                subgoal_achieved=False,
                progress_made=False,
                goal_achieved=False,
                suggest_retry=model_claimed_done,
            )

        # Use the most recent image for the VLM query
        latest = images[-1]
        full_goal_text = full_goal or current_subgoal

        prompt = f"""You are a goal verification system for a vision-language navigation agent.
Current subgoal to verify: "{current_subgoal}"
Full task goal: "{full_goal_text}"
The low-level controller reported "done" for the subgoal: {model_claimed_done}.

Based ONLY on the provided image (first-person / drone view), answer with a JSON object (no markdown, no extra text):
- "subgoal_achieved": true/false — Is the current subgoal clearly satisfied in this image?
- "progress_made": true/false — Is there visible progress toward the full goal?
- "goal_achieved": true/false — Is the full task goal satisfied?
- "suggest_retry": true/false — Suggest retry (e.g. if the model said done but the subgoal does not appear achieved).
"""

        try:
            response = self._llm.make_text_and_image_request(prompt, latest, temperature=0.0)
        except Exception as e:
            logger.error(f"Goal monitor VLM request failed: {e}")
            return GoalMonitorResult(
                subgoal_achieved=False,
                progress_made=False,
                goal_achieved=False,
                suggest_retry=model_claimed_done,
            )

        return self._parse_response(response, model_claimed_done)

    def _parse_response(self, response: str, model_claimed_done: bool) -> GoalMonitorResult:
        """Parse VLM response into GoalMonitorResult."""
        default = GoalMonitorResult(
            subgoal_achieved=False,
            progress_made=False,
            goal_achieved=False,
            suggest_retry=model_claimed_done,
        )
        response = response.strip()
        # Try to extract JSON from the response
        if "```" in response:
            for part in response.split("```"):
                part = part.strip()
                if part.startswith("json") or part.startswith("{"):
                    if part.startswith("json"):
                        part = part[4:].strip()
                    try:
                        d = json.loads(part)
                        return GoalMonitorResult(
                            subgoal_achieved=bool(d.get("subgoal_achieved", False)),
                            progress_made=bool(d.get("progress_made", False)),
                            goal_achieved=bool(d.get("goal_achieved", False)),
                            suggest_retry=bool(d.get("suggest_retry", model_claimed_done)),
                        )
                    except json.JSONDecodeError:
                        continue
        try:
            d = json.loads(response)
            return GoalMonitorResult(
                subgoal_achieved=bool(d.get("subgoal_achieved", False)),
                progress_made=bool(d.get("progress_made", False)),
                goal_achieved=bool(d.get("goal_achieved", False)),
                suggest_retry=bool(d.get("suggest_retry", model_claimed_done)),
            )
        except json.JSONDecodeError:
            logger.warning("Goal monitor could not parse JSON from response: %s", response[:200])
        return default
