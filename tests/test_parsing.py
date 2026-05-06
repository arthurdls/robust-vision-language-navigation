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


class TestParseLtlNlConstraints:
    def test_globally_not(self):
        pmap = {"pi_1": "go to A", "pi_2": "flying over building C"}
        result = parse_ltl_nl("G(!pi_2)", pmap)
        assert "ALWAYS" in result or "always" in result
        assert "flying over building C" in result

    def test_globally_bare(self):
        pmap = {"pi_1": "stay on course"}
        result = parse_ltl_nl("G pi_1", pmap)
        assert "ALWAYS" in result or "always" in result
        assert "stay on course" in result

    def test_formula_with_global_constraint(self):
        pmap = {
            "pi_1": "go to A",
            "pi_2": "go to B",
            "pi_3": "flying over building C",
        }
        result = parse_ltl_nl("F pi_2 & (!pi_2 U pi_1) & G(!pi_3)", pmap)
        assert "go to A" in result
        assert "go to B" in result
        assert "flying over building C" in result


# ---------------------------------------------------------------------------
# Comprehensive tests for find_main_operator_index
# ---------------------------------------------------------------------------

class TestFindMainOperatorComprehensive:
    """Exhaustive coverage of binary operator detection at top-level paren depth."""

    # --- Single operator at top level ---

    def test_single_and_at_top_level(self):
        op, idx = find_main_operator_index("A & B")
        assert op == "&"
        assert idx == 2

    def test_single_or_at_top_level(self):
        op, idx = find_main_operator_index("A | B")
        assert op == "|"
        assert idx == 2

    def test_single_until_at_top_level(self):
        op, idx = find_main_operator_index("A U B")
        assert op == "U"
        assert idx == 2

    # --- Precedence: & is returned over |, | over U ---

    def test_and_preferred_over_or(self):
        """& has lowest precedence, so it should be the main split point."""
        op, idx = find_main_operator_index("A | B & C | D")
        assert op == "&"

    def test_and_preferred_over_until(self):
        op, idx = find_main_operator_index("A U B & C U D")
        assert op == "&"

    def test_or_preferred_over_until(self):
        op, idx = find_main_operator_index("A U B | C U D")
        assert op == "|"

    def test_all_three_operators_at_top_level(self):
        """When &, |, and U all appear at top level, & should win."""
        op, idx = find_main_operator_index("A U B | C & D")
        assert op == "&"

    # --- Operators inside parentheses are skipped ---

    def test_operator_inside_parens_skipped(self):
        op, idx = find_main_operator_index("(A & B)")
        assert op is None
        assert idx == -1

    def test_inner_and_outer_or(self):
        op, idx = find_main_operator_index("(A & B) | C")
        assert op == "|"

    def test_deeply_nested_operators_invisible(self):
        op, idx = find_main_operator_index("((A & B) | (C U D))")
        assert op is None
        assert idx == -1

    def test_nested_parens_with_top_level_and(self):
        op, idx = find_main_operator_index("(A | B) & (C U D)")
        assert op == "&"

    # --- No operators ---

    def test_plain_predicate(self):
        op, idx = find_main_operator_index("pi_1")
        assert op is None
        assert idx == -1

    def test_unary_only_formula(self):
        op, idx = find_main_operator_index("F pi_1")
        assert op is None
        assert idx == -1

    def test_empty_string(self):
        op, idx = find_main_operator_index("")
        assert op is None
        assert idx == -1

    # --- Rightmost index is recorded ---

    def test_multiple_and_returns_rightmost(self):
        """When multiple & appear at top level, the rightmost one is stored."""
        op, idx = find_main_operator_index("A & B & C")
        assert op == "&"
        # "A & B & C"  indices: A=0, &=2, B=4, &=6, C=8
        assert idx == 6

    def test_multiple_or_returns_rightmost(self):
        op, idx = find_main_operator_index("A | B | C")
        assert op == "|"
        assert idx == 6

    # --- Complex realistic formulas ---

    def test_realistic_formula_with_constraint(self):
        """F pi_1 & (!pi_1 U pi_2) & G(!pi_3)"""
        formula = "F pi_1 & (!pi_1 U pi_2) & G(!pi_3)"
        op, idx = find_main_operator_index(formula)
        assert op == "&"

    def test_u_inside_parens_and_outside(self):
        """(A U B) & C U D -- & should win because it is lower precedence."""
        op, idx = find_main_operator_index("(A U B) & C U D")
        assert op == "&"


# ---------------------------------------------------------------------------
# Comprehensive tests for parse_ltl_nl
# ---------------------------------------------------------------------------

class TestParseLtlNlComprehensive:
    """Exhaustive coverage of the LTL-NL recursive parser."""

    # Shared predicate map used across tests
    PMAP = {
        "pi_1": "reach the building",
        "pi_2": "cross the bridge",
        "pi_3": "fly over the lake",
        "pi_4": "land on the rooftop",
        "pi_5": "pass the statue",
    }

    # -----------------------------------------------------------------------
    # 1. Base case: single predicate
    # -----------------------------------------------------------------------

    def test_single_predicate(self):
        result = parse_ltl_nl("pi_1", self.PMAP)
        assert result == "'reach the building'"

    def test_single_predicate_with_whitespace(self):
        result = parse_ltl_nl("  pi_2  ", self.PMAP)
        assert result == "'cross the bridge'"

    # -----------------------------------------------------------------------
    # 2. F (eventually) operator, both syntactic forms
    # -----------------------------------------------------------------------

    def test_eventually_space_form(self):
        result = parse_ltl_nl("F pi_1", self.PMAP)
        assert "eventually" in result
        assert "reach the building" in result
        assert "accomplished" in result

    def test_eventually_paren_form(self):
        result = parse_ltl_nl("F(pi_1)", self.PMAP)
        assert "eventually" in result
        assert "reach the building" in result

    def test_eventually_paren_with_compound_sub_expression(self):
        """F(pi_1 & pi_2) should parse the inner & expression."""
        result = parse_ltl_nl("F(pi_1 & pi_2)", self.PMAP)
        assert "eventually" in result
        assert "reach the building" in result
        assert "cross the bridge" in result
        assert "AND" in result

    # -----------------------------------------------------------------------
    # 3. G (globally/always) operator, both syntactic forms
    # -----------------------------------------------------------------------

    def test_globally_space_form(self):
        result = parse_ltl_nl("G pi_1", self.PMAP)
        assert "ALWAYS" in result
        assert "reach the building" in result

    def test_globally_paren_form(self):
        result = parse_ltl_nl("G(pi_1)", self.PMAP)
        assert "ALWAYS" in result
        assert "reach the building" in result

    def test_globally_with_negated_sub_expression(self):
        """G(!pi_3) is the standard negative constraint pattern."""
        result = parse_ltl_nl("G(!pi_3)", self.PMAP)
        assert "ALWAYS" in result
        assert "NOT" in result
        assert "fly over the lake" in result

    def test_globally_with_compound_sub_expression(self):
        """G(pi_1 | pi_2)"""
        result = parse_ltl_nl("G(pi_1 | pi_2)", self.PMAP)
        assert "ALWAYS" in result
        assert "OR" in result

    # -----------------------------------------------------------------------
    # 4. ! (negation) operator
    # -----------------------------------------------------------------------

    def test_negation_simple(self):
        result = parse_ltl_nl("!pi_1", self.PMAP)
        assert "NOT" in result
        assert "reach the building" in result

    def test_negation_of_parenthesized_expression(self):
        result = parse_ltl_nl("!(pi_1 & pi_2)", self.PMAP)
        assert "NOT" in result
        assert "AND" in result

    # -----------------------------------------------------------------------
    # 5. & (and) operator
    # -----------------------------------------------------------------------

    def test_and_two_predicates(self):
        result = parse_ltl_nl("pi_1 & pi_2", self.PMAP)
        assert "AND" in result
        assert "reach the building" in result
        assert "cross the bridge" in result
        assert "both" in result

    def test_and_two_eventually(self):
        result = parse_ltl_nl("F pi_1 & F pi_2", self.PMAP)
        assert "AND" in result
        assert "eventually" in result

    def test_and_three_way(self):
        """A & B & C splits on rightmost &, giving (A & B) on left, C on right."""
        result = parse_ltl_nl("pi_1 & pi_2 & pi_3", self.PMAP)
        assert "AND" in result
        assert "reach the building" in result
        assert "cross the bridge" in result
        assert "fly over the lake" in result

    # -----------------------------------------------------------------------
    # 6. | (or) operator
    # -----------------------------------------------------------------------

    def test_or_two_predicates(self):
        result = parse_ltl_nl("pi_1 | pi_2", self.PMAP)
        assert "OR" in result
        assert "reach the building" in result
        assert "cross the bridge" in result
        assert "either" in result

    def test_or_two_eventually(self):
        """Branching: F pi_1 | F pi_2"""
        result = parse_ltl_nl("F pi_1 | F pi_2", self.PMAP)
        assert "OR" in result
        assert "eventually" in result

    # -----------------------------------------------------------------------
    # 7. U (until) operator
    # -----------------------------------------------------------------------

    def test_until_non_negated_left(self):
        """pi_1 U pi_2: 'pi_1 must hold true UNTIL pi_2 is accomplished'"""
        result = parse_ltl_nl("pi_1 U pi_2", self.PMAP)
        assert "UNTIL" in result
        assert "reach the building" in result
        assert "cross the bridge" in result

    def test_until_negated_left_sequencing(self):
        """!pi_1 U pi_2: 'pi_2 must be accomplished BEFORE pi_1'"""
        result = parse_ltl_nl("!pi_1 U pi_2", self.PMAP)
        assert "BEFORE" in result
        assert "reach the building" in result
        assert "cross the bridge" in result

    def test_until_negated_left_different_predicates(self):
        result = parse_ltl_nl("!pi_3 U pi_4", self.PMAP)
        assert "BEFORE" in result
        assert "fly over the lake" in result
        assert "land on the rooftop" in result

    # -----------------------------------------------------------------------
    # 8. Parenthesized expressions (paren stripping)
    # -----------------------------------------------------------------------

    def test_parenthesized_single_predicate(self):
        result = parse_ltl_nl("(pi_1)", self.PMAP)
        assert result == "'reach the building'"

    def test_parenthesized_and_expression(self):
        result = parse_ltl_nl("(pi_1 & pi_2)", self.PMAP)
        assert "AND" in result

    def test_double_parenthesized_expression(self):
        result = parse_ltl_nl("((pi_1))", self.PMAP)
        assert result == "'reach the building'"

    # -----------------------------------------------------------------------
    # 9. Nested unary operators
    # -----------------------------------------------------------------------

    def test_f_of_negation(self):
        """F !pi_1"""
        result = parse_ltl_nl("F !pi_1", self.PMAP)
        assert "eventually" in result
        assert "NOT" in result

    def test_g_of_f(self):
        """G(F pi_1) -- nested G around F"""
        # G( starts the G-paren path, inner is F pi_1
        result = parse_ltl_nl("G(F pi_1)", self.PMAP)
        assert "ALWAYS" in result
        assert "eventually" in result
        assert "reach the building" in result

    def test_f_of_g(self):
        """F(G pi_1)"""
        result = parse_ltl_nl("F(G pi_1)", self.PMAP)
        assert "eventually" in result
        assert "ALWAYS" in result

    def test_double_negation(self):
        """!!pi_1"""
        result = parse_ltl_nl("!!pi_1", self.PMAP)
        assert result.count("NOT") == 2

    # -----------------------------------------------------------------------
    # 10. Complex realistic formulas
    # -----------------------------------------------------------------------

    def test_sequential_with_constraint(self):
        """F pi_1 & (!pi_1 U pi_2) & G(!pi_3)
        - eventually reach the building
        - cross the bridge before reaching the building
        - always avoid flying over the lake
        """
        formula = "F pi_1 & (!pi_1 U pi_2) & G(!pi_3)"
        result = parse_ltl_nl(formula, self.PMAP)
        assert "reach the building" in result
        assert "cross the bridge" in result
        assert "fly over the lake" in result
        assert "AND" in result

    def test_sequential_with_constraint_parenthesized(self):
        """(F pi_1 & (!pi_1 U pi_2)) & G(!pi_3)"""
        formula = "(F pi_1 & (!pi_1 U pi_2)) & G(!pi_3)"
        result = parse_ltl_nl(formula, self.PMAP)
        assert "reach the building" in result
        assert "cross the bridge" in result
        assert "fly over the lake" in result
        assert "AND" in result
        assert "ALWAYS" in result

    def test_multiple_global_negations(self):
        """G(!pi_3) & G(!pi_4)"""
        result = parse_ltl_nl("G(!pi_3) & G(!pi_4)", self.PMAP)
        assert "AND" in result
        assert "ALWAYS" in result
        assert "fly over the lake" in result
        assert "land on the rooftop" in result

    def test_branching_with_eventually(self):
        """F pi_1 | F pi_2"""
        result = parse_ltl_nl("F pi_1 | F pi_2", self.PMAP)
        assert "OR" in result
        assert "reach the building" in result
        assert "cross the bridge" in result

    def test_triple_sequence_via_and(self):
        """(!pi_1 U pi_2) & (!pi_2 U pi_3)
        Represents: first reach pi_2, then reach pi_3 (pi_2 before pi_1, pi_3 before pi_2).
        """
        result = parse_ltl_nl("(!pi_1 U pi_2) & (!pi_2 U pi_3)", self.PMAP)
        assert "AND" in result
        assert "BEFORE" in result
        assert "reach the building" in result
        assert "cross the bridge" in result
        assert "fly over the lake" in result

    def test_eventually_and_globally_combined(self):
        """F pi_1 & G pi_2"""
        result = parse_ltl_nl("F pi_1 & G pi_2", self.PMAP)
        assert "AND" in result
        assert "eventually" in result
        assert "ALWAYS" in result

    def test_or_with_constraints(self):
        """(F pi_1 | F pi_2) & G(!pi_3)"""
        result = parse_ltl_nl("(F pi_1 | F pi_2) & G(!pi_3)", self.PMAP)
        assert "AND" in result
        assert "OR" in result
        assert "ALWAYS" in result
        assert "NOT" in result

    def test_deeply_nested_formula(self):
        """((F pi_1 & (!pi_1 U pi_2)) & (!pi_2 U pi_3)) & G(!pi_4)"""
        formula = "((F pi_1 & (!pi_1 U pi_2)) & (!pi_2 U pi_3)) & G(!pi_4)"
        result = parse_ltl_nl(formula, self.PMAP)
        assert "reach the building" in result
        assert "cross the bridge" in result
        assert "fly over the lake" in result
        assert "land on the rooftop" in result
        assert "AND" in result

    def test_until_with_eventually_right(self):
        """pi_1 U F pi_2 -- hold pi_1 until eventually pi_2"""
        result = parse_ltl_nl("pi_1 U F pi_2", self.PMAP)
        assert "UNTIL" in result
        assert "reach the building" in result
        assert "eventually" in result

    # -----------------------------------------------------------------------
    # 11. Positive constraint pattern: G(pi_X)
    # -----------------------------------------------------------------------

    def test_positive_constraint(self):
        """G(pi_1) -- always maintain this condition."""
        result = parse_ltl_nl("G(pi_1)", self.PMAP)
        assert "ALWAYS" in result
        assert "reach the building" in result
        # Should NOT contain "NOT"
        assert "NOT" not in result

    # -----------------------------------------------------------------------
    # 12. Error handling
    # -----------------------------------------------------------------------

    def test_unknown_formula_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_ltl_nl("UNKNOWN_TOKEN", {})

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_ltl_nl("", {})

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_ltl_nl("   ", {})

    def test_undefined_predicate_raises_value_error(self):
        """A predicate that is not in the map and has no operator structure."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_ltl_nl("pi_99", self.PMAP)

    def test_malformed_binary_formula_raises(self):
        """Operator with missing right operand."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_ltl_nl("pi_1 &", self.PMAP)

    # -----------------------------------------------------------------------
    # 13. Verify specific output substrings for each operator
    # -----------------------------------------------------------------------

    def test_and_output_format(self):
        result = parse_ltl_nl("pi_1 & pi_2", self.PMAP)
        assert "AND" in result
        assert "both be accomplished" in result

    def test_or_output_format(self):
        result = parse_ltl_nl("pi_1 | pi_2", self.PMAP)
        assert "either" in result
        assert "OR" in result
        assert "accomplished" in result

    def test_until_output_format(self):
        result = parse_ltl_nl("pi_1 U pi_2", self.PMAP)
        assert "hold true UNTIL" in result
        assert "accomplished" in result

    def test_until_negated_output_format(self):
        result = parse_ltl_nl("!pi_1 U pi_2", self.PMAP)
        assert "BEFORE" in result
        assert "accomplished" in result

    def test_eventually_output_format(self):
        result = parse_ltl_nl("F pi_1", self.PMAP)
        assert result.startswith("(eventually")
        assert result.endswith("accomplished)")

    def test_globally_output_format(self):
        result = parse_ltl_nl("G pi_1", self.PMAP)
        assert "ALWAYS be the case" in result

    def test_negation_output_format(self):
        result = parse_ltl_nl("!pi_1", self.PMAP)
        assert "NOT be the case" in result

    # -----------------------------------------------------------------------
    # 14. Edge cases and whitespace handling
    # -----------------------------------------------------------------------

    def test_extra_spaces_around_binary_op(self):
        result = parse_ltl_nl("pi_1  &  pi_2", self.PMAP)
        assert "AND" in result
        assert "reach the building" in result
        assert "cross the bridge" in result

    def test_formula_with_leading_trailing_whitespace(self):
        result = parse_ltl_nl("  F pi_1  ", self.PMAP)
        assert "eventually" in result
        assert "reach the building" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
