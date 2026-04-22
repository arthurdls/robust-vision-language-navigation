"""Tests for rvln.ai.utils.parsing -- extract_json and LTL-NL parsing."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from rvln.ai.utils.parsing import extract_json, parse_ltl_nl, find_main_operator_index


class TestExtractJson:
    def test_plain_json(self):
        result = extract_json('{"complete": true, "score": 0.8}')
        assert result == {"complete": True, "score": 0.8}

    def test_markdown_fence(self):
        text = '```json\n{"complete": false, "score": 0.3}\n```'
        assert extract_json(text) == {"complete": False, "score": 0.3}

    def test_markdown_fence_no_lang(self):
        text = '```\n{"val": 42}\n```'
        assert extract_json(text) == {"val": 42}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"done": true}\nEnd of response.'
        assert extract_json(text) == {"done": True}

    def test_empty_input_raises(self):
        with pytest.raises(SyntaxError, match="empty input"):
            extract_json("")

    def test_no_json_raises(self):
        with pytest.raises(SyntaxError, match="no opening brace"):
            extract_json("no json here")

    def test_malformed_json_raises(self):
        with pytest.raises(SyntaxError, match="not parseable"):
            extract_json("{bad json}")

    def test_nested_json(self):
        text = '{"outer": {"inner": 1}}'
        assert extract_json(text) == {"outer": {"inner": 1}}

    def test_fence_with_whitespace(self):
        text = '```json\n  {"x": 1}  \n```'
        assert extract_json(text) == {"x": 1}

    def test_brace_fallback_with_leading_text(self):
        text = 'The answer is {"key": "value"} as expected.'
        assert extract_json(text) == {"key": "value"}


class TestFindMainOperator:
    def test_and_operator(self):
        op, idx = find_main_operator_index("F pi_1 & F pi_2")
        assert op == "&"

    def test_nested_parens_skipped(self):
        op, idx = find_main_operator_index("(A & B) | C")
        assert op == "|"

    def test_no_operator(self):
        op, idx = find_main_operator_index("pi_1")
        assert op is None
        assert idx == -1


class TestParseLtlNl:
    def test_single_predicate(self):
        result = parse_ltl_nl("pi_1", {"pi_1": "reach the building"})
        assert result == "'reach the building'"

    def test_eventually(self):
        result = parse_ltl_nl("F pi_1", {"pi_1": "reach the building"})
        assert "eventually" in result
        assert "reach the building" in result

    def test_sequence(self):
        pmap = {"pi_1": "go to A", "pi_2": "go to B"}
        result = parse_ltl_nl("!pi_1 U pi_2", pmap)
        assert "BEFORE" in result

    def test_unknown_formula_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_ltl_nl("UNKNOWN", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
