"""
Unit tests for SubgoalConverter._parse_response.

The SubgoalConverter translates LTL planner predicates into short OpenVLA
instructions. _parse_response validates the LLM response and returns a
(instruction_or_None, error_or_None) pair so the public `convert()` can
retry once on validation failure rather than silently passing raw text
through to OpenVLA.

No API calls: all tests exercise the parsing logic directly.
Run: conda run -n rvln-sim pytest tests/test_subgoal_converter.py -v
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.ai.subgoal_converter import SubgoalConverter, ConversionResult


class TestParseResponseValidJson:
    def test_normal_instruction(self):
        instruction, error = SubgoalConverter._parse_response('{"sub_goal": "turn right"}')
        assert instruction == "turn right"
        assert error is None

    def test_approach_command(self):
        instruction, error = SubgoalConverter._parse_response(
            '{"sub_goal": "approach the tree"}',
        )
        assert instruction == "approach the tree"
        assert error is None

    def test_whitespace_in_sub_goal_stripped(self):
        instruction, error = SubgoalConverter._parse_response(
            '{"sub_goal": "  move forward  "}',
        )
        assert instruction == "move forward"
        assert error is None

    def test_json_with_extra_fields(self):
        instruction, error = SubgoalConverter._parse_response(
            '{"sub_goal": "ascend 5 meters", "confidence": 0.9}',
        )
        assert instruction == "ascend 5 meters"
        assert error is None

    def test_code_fence_tolerated(self):
        instruction, error = SubgoalConverter._parse_response(
            '```json\n{"sub_goal": "turn left"}\n```',
        )
        assert instruction == "turn left"
        assert error is None


class TestParseResponseRejection:
    def test_missing_sub_goal_field(self):
        instruction, error = SubgoalConverter._parse_response('{}')
        assert instruction is None
        assert error and "sub_goal" in error

    def test_empty_sub_goal_string(self):
        instruction, error = SubgoalConverter._parse_response('{"sub_goal": ""}')
        assert instruction is None
        assert error and "empty" in error

    def test_whitespace_only_sub_goal(self):
        instruction, error = SubgoalConverter._parse_response('{"sub_goal": "   "}')
        assert instruction is None
        assert error and "empty" in error

    def test_non_string_sub_goal(self):
        instruction, error = SubgoalConverter._parse_response('{"sub_goal": 42}')
        assert instruction is None
        assert error is not None

    def test_plain_text_rejected(self):
        # Bare text is no longer silently accepted: it produces a JSON
        # decode error so convert() can retry with the error in the prompt.
        instruction, error = SubgoalConverter._parse_response("turn right")
        assert instruction is None
        assert error and "JSON" in error

    def test_malformed_json_rejected(self):
        instruction, error = SubgoalConverter._parse_response('{"sub_goal": "turn right"')
        assert instruction is None
        assert error and "JSON" in error

    def test_empty_string_rejected(self):
        instruction, error = SubgoalConverter._parse_response("")
        assert instruction is None
        assert error is not None

    def test_oversized_instruction_rejected(self):
        long_instr = "x" * (SubgoalConverter.MAX_INSTRUCTION_CHARS + 1)
        instruction, error = SubgoalConverter._parse_response(
            f'{{"sub_goal": "{long_instr}"}}'
        )
        assert instruction is None
        assert error and "too long" in error


class TestConversionResultDataclass:
    def test_fields(self):
        r = ConversionResult(instruction="go forward")
        assert r.instruction == "go forward"

    def test_equality(self):
        r1 = ConversionResult(instruction="a")
        r2 = ConversionResult(instruction="a")
        assert r1 == r2
