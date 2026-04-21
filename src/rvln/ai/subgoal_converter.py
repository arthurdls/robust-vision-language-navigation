"""
Subgoal Converter: translates natural language subgoals into short,
imperative OpenVLA-compatible instructions.

OpenVLA was trained on short drone commands.  This layer strips stopping
conditions and extracts the core physical action so the model receives
an instruction it can actually execute.

Called once per task at the start of a run (not every step).
"""

import logging
from typing import Optional

from ..config import DEFAULT_VLM_MODEL
from .prompts import SUBGOAL_CONVERSION_PROMPT as CONVERSION_SYSTEM_PROMPT
from .utils.llm_providers import BaseLLM, LLMFactory

logger = logging.getLogger(__name__)


class SubgoalConverter:
    """Converts a natural language subgoal into an OpenVLA-compatible instruction.

    OpenVLA was trained on short imperative drone commands.  This layer strips
    stopping conditions and extracts the core physical action.
    """

    def __init__(self, model: str = DEFAULT_VLM_MODEL):
        self._model = model
        self._llm: BaseLLM = self._make_llm(model)

    def convert(self, subgoal: str) -> str:
        """Convert *subgoal* to an OpenVLA instruction string."""
        messages = [
            {"role": "system", "content": CONVERSION_SYSTEM_PROMPT},
            {"role": "user", "content": subgoal},
        ]
        response = self._llm.make_request(messages, temperature=0.0)
        instruction = response.strip().strip('"').strip("'")
        logger.info("SubgoalConverter: '%s' -> '%s'", subgoal, instruction)
        return instruction

    @staticmethod
    def _make_llm(model: str) -> BaseLLM:
        if model.startswith("gemini"):
            return LLMFactory.create("gemini", model=model)
        return LLMFactory.create("openai", model=model)
