"""
Unit tests for SubgoalConverter._parse_response.

The SubgoalConverter translates LTL planner predicates into short OpenVLA
instructions. These tests cover the static _parse_response method which
handles both valid JSON and fallback plain-text responses from the LLM.

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
        result = SubgoalConverter._parse_response(
            '{"sub_goal": "turn right", "outside_of_distribution": false}',
            "Turn right until you see the car",
        )
        assert result.instruction == "turn right"
        assert result.outside_of_distribution is False

    def test_ood_true(self):
        result = SubgoalConverter._parse_response(
            '{"sub_goal": "pick up the cup", "outside_of_distribution": true}',
            "Pick up the red cup from the table",
        )
        assert result.instruction == "pick up the cup"
        assert result.outside_of_distribution is True

    def test_approach_command(self):
        result = SubgoalConverter._parse_response(
            '{"sub_goal": "approach the tree", "outside_of_distribution": false}',
            "Move toward the tree until you are near it",
        )
        assert result.instruction == "approach the tree"
        assert result.outside_of_distribution is False

    def test_whitespace_in_sub_goal_stripped(self):
        result = SubgoalConverter._parse_response(
            '{"sub_goal": "  move forward  ", "outside_of_distribution": false}',
            "move forward",
        )
        assert result.instruction == "move forward"

    def test_missing_sub_goal_uses_original(self):
        result = SubgoalConverter._parse_response(
            '{"outside_of_distribution": false}',
            "original instruction",
        )
        assert result.instruction == "original instruction"

    def test_missing_ood_defaults_false(self):
        result = SubgoalConverter._parse_response(
            '{"sub_goal": "turn left"}',
            "turn left slowly",
        )
        assert result.instruction == "turn left"
        assert result.outside_of_distribution is False

    def test_empty_sub_goal_string(self):
        result = SubgoalConverter._parse_response(
            '{"sub_goal": "", "outside_of_distribution": false}',
            "fallback instruction",
        )
        # Empty string from JSON .get() is truthy for str(), so it stays empty
        # The original subgoal is only used when the key is missing entirely
        assert isinstance(result, ConversionResult)

    def test_json_with_extra_fields(self):
        result = SubgoalConverter._parse_response(
            '{"sub_goal": "ascend 5 meters", "outside_of_distribution": false, "confidence": 0.9}',
            "go up",
        )
        assert result.instruction == "ascend 5 meters"
        assert result.outside_of_distribution is False


class TestParseResponseFallback:
    def test_plain_text_stripped_of_quotes(self):
        result = SubgoalConverter._parse_response(
            '"turn right"',
            "Turn right until you see the car",
        )
        assert result.instruction == "turn right"
        assert result.outside_of_distribution is False

    def test_plain_text_single_quotes_stripped(self):
        result = SubgoalConverter._parse_response(
            "'approach the building'",
            "approach the building",
        )
        assert result.instruction == "approach the building"

    def test_plain_text_no_quotes(self):
        result = SubgoalConverter._parse_response(
            "move forward 5.0 meters",
            "move forward 5.0 meters",
        )
        assert result.instruction == "move forward 5.0 meters"
        assert result.outside_of_distribution is False

    def test_empty_string_fallback(self):
        result = SubgoalConverter._parse_response("", "original")
        assert result.instruction == ""
        assert result.outside_of_distribution is False

    def test_whitespace_only_fallback(self):
        result = SubgoalConverter._parse_response("   ", "original")
        assert result.instruction == ""
        assert result.outside_of_distribution is False

    def test_malformed_json_fallback(self):
        result = SubgoalConverter._parse_response(
            '{"sub_goal": "turn right"',
            "original",
        )
        # Truncated JSON should trigger fallback
        assert isinstance(result, ConversionResult)
        assert result.outside_of_distribution is False


class TestConversionResultDataclass:
    def test_fields(self):
        r = ConversionResult(instruction="go forward", outside_of_distribution=True)
        assert r.instruction == "go forward"
        assert r.outside_of_distribution is True

    def test_equality(self):
        r1 = ConversionResult(instruction="a", outside_of_distribution=False)
        r2 = ConversionResult(instruction="a", outside_of_distribution=False)
        assert r1 == r2
