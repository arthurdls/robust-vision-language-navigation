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

from .utils.base_llm_providers import BaseLLM, LLMFactory

logger = logging.getLogger(__name__)


CONVERSION_SYSTEM_PROMPT = """\
You convert natural language drone subgoals into short, imperative instructions
that a vision-language-action model (OpenVLA) can execute. OpenVLA understands
commands like:
- "turn right", "turn left 90 degrees"
- "move forward", "move forward 10 meters"
- "go between the tree and the streetlight"
- "move above the pergola", "descend 5 meters"
- "approach the building", "get close to the person"

Rules:
- Strip any stopping/completion conditions (e.g., "until you see X" -> just the action)
- Keep spatial references that help the model navigate (e.g., "between X and Y")
- Output ONLY the instruction string, nothing else.

Examples:
  "Turn right until you see the red car" -> "turn right"
  "Continue forward until close to the person ahead" -> "move forward"
  "Go between the tree and the streetlight" -> "go between the tree and the streetlight"
  "Move through the pergola between the wooden poles" -> "move through the pergola"
  "Go above the pergola" -> "go above the pergola"
  "Approach the white building" -> "approach the white building"\
"""


class SubgoalConverter:
    """Converts a natural language subgoal into an OpenVLA-compatible instruction.

    OpenVLA was trained on short imperative drone commands.  This layer strips
    stopping conditions and extracts the core physical action.
    """

    def __init__(self, model: str = "gpt-4o"):
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
