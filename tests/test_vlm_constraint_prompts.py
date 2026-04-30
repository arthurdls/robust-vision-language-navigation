"""
Tier 3 VLM prompt smoke tests for constraint-aware diary monitoring.

These tests make real VLM API calls with synthetic diary context to verify
the VLM correctly returns constraint_violated in its JSON response.

Run: conda run -n rvln-sim pytest tests/test_vlm_constraint_prompts.py -v -m "tier3"
"""
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

needs_api = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="No OPENAI_API_KEY",
)
tier3 = pytest.mark.tier3


def _render_global_prompt(subgoal, constraints, diary, displacement, prev_pct):
    from rvln.ai.prompts import DIARY_GLOBAL_PROMPT

    constraints_block = ""
    if constraints:
        lines = ["Active constraints (must be maintained throughout):"]
        for c in constraints:
            lines.append(f"  - {c}")
        lines.append("")
        constraints_block = "\n".join(lines)

    return DIARY_GLOBAL_PROMPT.format(
        subgoal=subgoal,
        constraints_block=constraints_block,
        diary=diary,
        prev_completion_pct=prev_pct,
        displacement=displacement,
    )


@needs_api
@tier3
def test_vlm_returns_constraint_violated_field():
    """VLM should include constraint_violated in JSON when constraints are present."""
    from rvln.ai.prompts import DIARY_SYSTEM_PROMPT
    from rvln.ai.utils.llm_providers import LLMFactory

    llm = LLMFactory.create("openai", model="gpt-4o")

    prompt = _render_global_prompt(
        subgoal="Approach the tree",
        constraints=["stay away from building B"],
        diary="Steps 0-20: Drone moved forward, building B visible on the right.\n"
              "Steps 20-40: Drone turned slightly toward building B.\n"
              "Checkpoint 40: completion = 0.30",
        displacement="[x: 3.50 m, y: 1.20 m, z: 0.00 m, yaw: 15.0 deg]",
        prev_pct=0.30,
    )

    messages = [
        {"role": "system", "content": DIARY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    response = llm.make_request(messages, temperature=0.0)
    text = response.strip()

    start = text.find("{")
    end = text.rfind("}")
    assert start != -1 and end != -1, f"No JSON in VLM response: {text}"
    parsed = json.loads(text[start:end + 1])

    assert "constraint_violated" in parsed, (
        f"VLM response missing 'constraint_violated' field. Got: {parsed}"
    )
    assert isinstance(parsed["constraint_violated"], bool), (
        f"'constraint_violated' should be bool, got: {type(parsed['constraint_violated'])}"
    )


