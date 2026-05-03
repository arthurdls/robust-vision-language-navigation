"""
Subgoal Converter: translates natural language subgoals into short,
imperative OpenVLA-compatible instructions.

OpenVLA was trained on short drone commands.  This layer strips stopping
conditions and extracts the core physical action so the model receives
an instruction it can actually execute.

Called once per task at the start of a run (not every step).
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_VLM_MODEL
from .prompts import SUBGOAL_CONVERSION_PROMPT as CONVERSION_SYSTEM_PROMPT
from .utils.llm_providers import BaseLLM, LLMFactory

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Structured output from the subgoal converter."""
    instruction: str


class SubgoalConverter:
    """Converts a natural language subgoal into an OpenVLA-compatible instruction.

    OpenVLA was trained on short imperative drone commands.  This layer strips
    stopping conditions and extracts the core physical action.
    """

    def __init__(self, model: str = DEFAULT_VLM_MODEL):
        self._model = model
        self._llm: BaseLLM = self._make_llm(model)
        self.llm_call_records: List[Dict[str, Any]] = []

    def convert(self, subgoal: str) -> ConversionResult:
        """Convert *subgoal* to a ConversionResult with the translated instruction."""
        messages = [
            {"role": "system", "content": CONVERSION_SYSTEM_PROMPT},
            {"role": "user", "content": subgoal},
        ]
        t0 = time.time()
        response = self._llm.make_request(messages, temperature=0.0)
        rtt = time.time() - t0
        usage = self._llm.last_usage
        self.llm_call_records.append({
            "label": "subgoal_convert",
            "subgoal": subgoal,
            "rtt_s": round(rtt, 3),
            "model": usage.get("model", self._model),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })

        result = self._parse_response(response, subgoal)
        logger.info(
            "SubgoalConverter: '%s' -> '%s'",
            subgoal, result.instruction,
        )
        return result

    @staticmethod
    def _parse_response(response: str, original_subgoal: str) -> ConversionResult:
        text = response.strip()
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise TypeError(f"Expected JSON object, got {type(parsed).__name__}")
            return ConversionResult(
                instruction=str(parsed.get("sub_goal", original_subgoal)).strip(),
            )
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            logger.warning(
                "SubgoalConverter: failed to parse JSON, falling back to plain text: %s",
                text,
            )
            instruction = text.strip('"').strip("'")
            return ConversionResult(instruction=instruction)

    @staticmethod
    def _make_llm(model: str) -> BaseLLM:
        if model.startswith("gemini"):
            return LLMFactory.create("gemini", model=model)
        return LLMFactory.create("openai", model=model)
