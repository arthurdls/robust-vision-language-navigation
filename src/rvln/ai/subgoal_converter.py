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

from ..config import DEFAULT_LLM_MODEL
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

    def __init__(self, model: str = DEFAULT_LLM_MODEL):
        self._model = model
        self._llm: BaseLLM = self._make_llm(model)
        self.llm_call_records: List[Dict[str, Any]] = []

    # Hard cap on the converted instruction length. OpenVLA was trained on
    # short imperative drone commands; passing through a multi-paragraph
    # LLM hedge as the instruction overflows that distribution and produces
    # undefined actions. 200 chars is well above any well-formed converted
    # instruction observed during development.
    MAX_INSTRUCTION_CHARS = 200

    def convert(self, subgoal: str) -> ConversionResult:
        """Convert *subgoal* to a ConversionResult with the translated instruction.

        Tries up to two LLM calls: the first gets the system prompt; if the
        response cannot be parsed as a {"sub_goal": str} JSON object or the
        extracted instruction is empty / oversized, the second call appends
        the parse error to the user prompt so the model can self-correct.
        Raises RuntimeError if both attempts fail. This is preferred over
        silently passing raw LLM text through to OpenVLA.
        """
        # First attempt
        instruction, parse_error = self._attempt_convert(subgoal, retry_hint=None)
        if instruction is not None:
            return ConversionResult(instruction=instruction)

        # Retry with the parse error in the user prompt.
        logger.warning(
            "SubgoalConverter: first attempt unparseable for '%s' (%s), retrying.",
            subgoal, parse_error,
        )
        instruction, parse_error = self._attempt_convert(
            subgoal, retry_hint=parse_error,
        )
        if instruction is not None:
            return ConversionResult(instruction=instruction)

        # Both attempts failed: surface as a hard error so the runner can
        # fail loudly instead of feeding garbage to OpenVLA.
        raise RuntimeError(
            f"SubgoalConverter could not produce a valid instruction for "
            f"subgoal '{subgoal}' after retry: {parse_error}"
        )

    def _attempt_convert(
        self, subgoal: str, retry_hint: Optional[str],
    ) -> tuple:
        """Run a single conversion call. Returns (instruction_or_None, error_or_None)."""
        user_content = subgoal
        label = "subgoal_convert"
        if retry_hint is not None:
            user_content = (
                f"{subgoal}\n\n"
                f"Previous attempt failed validation: {retry_hint}\n"
                "Respond with EXACTLY one JSON object: "
                '{"sub_goal": "<short imperative instruction>"}'
            )
            label = "subgoal_convert_retry"
        messages = [
            {"role": "system", "content": CONVERSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        t0 = time.time()
        response = self._llm.make_request(messages, temperature=0.0)
        rtt = time.time() - t0
        usage = self._llm.last_usage
        self.llm_call_records.append({
            "label": label,
            "subgoal": subgoal,
            "rtt_s": round(rtt, 3),
            "model": usage.get("model", self._model),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })
        instruction, error = self._parse_response(response)
        if instruction is not None:
            logger.info(
                "SubgoalConverter: '%s' -> '%s'",
                subgoal, instruction,
            )
        return instruction, error

    @classmethod
    def _parse_response(cls, response: str) -> tuple:
        """Parse a converter response. Returns (instruction_or_None, error_or_None).

        Validates: response is a JSON object with a non-empty string `sub_goal`
        field, length within MAX_INSTRUCTION_CHARS. Anything else is an error.
        """
        text = response.strip()
        # Tolerate ```json fences.
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"JSON decode error: {exc}"
        if not isinstance(parsed, dict):
            return None, f"expected JSON object, got {type(parsed).__name__}"
        raw = parsed.get("sub_goal")
        if not isinstance(raw, str):
            return None, "missing string field 'sub_goal'"
        instruction = raw.strip()
        if not instruction:
            return None, "empty 'sub_goal'"
        if len(instruction) > cls.MAX_INSTRUCTION_CHARS:
            return None, (
                f"instruction too long ({len(instruction)} > "
                f"{cls.MAX_INSTRUCTION_CHARS} chars)"
            )
        return instruction, None

    @staticmethod
    def _make_llm(model: str) -> BaseLLM:
        return LLMFactory.create("openai", model=model)
