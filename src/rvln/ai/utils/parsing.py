import json

def find_main_operator_index(formula):
    """
    Finds the main binary operator in the formula that is not enclosed in parentheses.
    Operators are checked in order of increasing precedence: &, |, U.
    """
    paren_level = 0
    # Store the rightmost index of each top-level operator found
    split_indices = {'&': -1, '|': -1, 'U': -1}

    for i, char in enumerate(formula):
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level -= 1
        elif paren_level == 0 and char in split_indices:
            # This operator is at the top level (not inside any parentheses)
            split_indices[char] = i

    # Return the operator with the lowest precedence found
    if split_indices['&'] != -1:
        return '&', split_indices['&']
    if split_indices['|'] != -1:
        return '|', split_indices['|']
    if split_indices['U'] != -1:
        return 'U', split_indices['U']

    return None, -1


def parse_ltl_nl(formula: str, predicate_map: dict[str, str]) -> str:
    """
    Recursively parses an LTL-NL formula string into a human-readable format.

    Args:
        formula: The LTL-NL formula string (e.g., "F pi_1 & (!pi_1 U pi_2)").
        predicate_map: A dictionary mapping predicate IDs to their text descriptions.

    Returns:
        A clarified, human-readable string explaining the formula.
    """
    formula = formula.strip()

    # Base Case: The formula is a single, defined predicate.
    if formula in predicate_map:
        return f"'{predicate_map[formula]}'"

    # --- Recursive Step ---

    # 1. Check for a top-level binary operator (&, |, U)
    op, idx = find_main_operator_index(formula)
    if op:
        left = formula[:idx].strip()
        right = formula[idx+1:].strip()

        # Recurse on the left and right sides
        parsed_left = parse_ltl_nl(left, predicate_map)
        parsed_right = parse_ltl_nl(right, predicate_map)

        if op == '&':
            return f"({parsed_left} AND {parsed_right} must both be accomplished in any order)"
        elif op == '|':
            return f"(either {parsed_left} OR {parsed_right} must be accomplished in any order)"
        elif op == 'U':
            # Handle the common sequencing pattern "!A U B" as "B before A"
            if left.startswith('!'):
                negated_predicate_text = parse_ltl_nl(left[1:].strip(), predicate_map)
                return f"({parsed_right} must be accomplished BEFORE {negated_predicate_text})"
            else:
                return f"({parsed_left} must hold true UNTIL {parsed_right} is accomplished)"

    # 2. If no binary operator, check for unary operators (F, !)
    if formula.startswith('F '):
        sub_formula = formula[1:].strip()
        return f"(eventually {parse_ltl_nl(sub_formula, predicate_map)} must be accomplished)"

    if formula.startswith('G ') or formula.startswith('G('):
        sub_formula = formula[2:].strip() if formula.startswith('G ') else formula[1:].strip()
        return f"(it must ALWAYS be the case that {parse_ltl_nl(sub_formula, predicate_map)})"

    if formula.startswith('!'):
        sub_formula = formula[1:].strip()
        return f"(it must NOT be the case that {parse_ltl_nl(sub_formula, predicate_map)})"

    # 3. If no operators, check for an expression fully enclosed in parentheses
    if formula.startswith('(') and formula.endswith(')'):
        # Recurse on the content inside the parentheses
        return parse_ltl_nl(formula[1:-1], predicate_map)

    # If the formula format is unrecognized, raise an error.
    raise ValueError(f"Could not parse formula or sub-formula: '{formula}'")


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a blob of text.

    Handles markdown code fences (```json ... ```) and raw JSON.
    Returns the parsed dict or raises SyntaxError.
    """
    if not text:
        raise SyntaxError("JSON text not parseable: empty input")

    import re
    # Try markdown code fence first
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try parsing the full text as JSON
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Fall back to finding the outermost braces
    start = text.find("{")
    if start == -1:
        raise SyntaxError("JSON text not parseable: no opening brace found")
    end = text.rfind("}")
    if end == -1 or end < start:
        raise SyntaxError("JSON text not parseable: no valid closing brace found")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise SyntaxError(f"JSON text not parseable: {e}") from e