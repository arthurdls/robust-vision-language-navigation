"""
LTL Symbolic Planner: parses natural language to LTL-NL via LLM, then uses Spot
to manage the automaton state and determine the next short-horizon subgoal.
"""

from typing import Optional

import spot

from .llm_interface import LLMUserInterface


def _predicate_key_to_index(key: str) -> int:
    """Parse predicate key to integer index (e.g. 'pi_1' -> 1, 'p1' -> 1)."""
    key = key.strip()
    if key.startswith("pi_"):
        return int(key[3:].strip())
    if key.startswith("p") and key[1:].strip().isdigit():
        return int(key[1:].strip())
    raise ValueError(f"Predicate key must be 'pi_N' or 'pN': got '{key}'")


def _normalize_pi_predicates(raw: dict) -> dict:
    """Normalize pi_predicates to canonical keys pi_1, pi_2, ... (ordered by index)."""
    if not raw or not isinstance(raw, dict):
        return {}
    result = {}
    for k, v in raw.items():
        if not isinstance(v, str):
            continue
        try:
            idx = _predicate_key_to_index(k)
        except (ValueError, TypeError):
            continue
        result[f"pi_{idx}"] = v.strip()
    return dict(sorted(result.items(), key=lambda x: _predicate_key_to_index(x[0])))


class LTLSymbolicPlanner:
    """
    Integrates LLMUserInterface to parse NL to LTL-NL, then uses Spot
    to manage the automaton state and determine the next action.
    """

    def __init__(self, llm_interface: LLMUserInterface):
        self.llm_interface = llm_interface
        self.current_automaton_state = 0
        self.automaton = None
        self._sink_state: Optional[int] = None  # Terminal state added by _add_sink_state
        self.pi_map = {}  # Maps pi_x -> Natural Language description
        self._last_returned_predicate_key: Optional[str] = None  # Key of predicate last returned by get_next_predicate (allows duplicate descriptions)
        self.finished = False
        self._raw_formula: str = ""
        self.constraint_predicates: dict[str, str] = {}

    def plan_from_natural_language(self, instruction: str) -> None:
        """
        1. Query LLM for LTL formula.
        2. Convert LTL string to Spot Automaton.
        """
        if not instruction or not isinstance(instruction, str):
            raise ValueError("Instruction must be a non-empty string.")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Instruction must be a non-empty string.")

        print(f"[LTL Planner] Processing instruction: '{instruction}'")
        _ = self.llm_interface.make_natural_language_request(instruction)
        data = self.llm_interface.ltl_nl_formula

        if not data or not isinstance(data, dict):
            raise ValueError("LLM could not generate a valid LTL formula (empty or non-dict response).")
        if "ltl_nl_formula" not in data or "pi_predicates" not in data:
            raise ValueError(
                "LLM response must contain 'ltl_nl_formula' and 'pi_predicates'. "
                f"Keys received: {list(data.keys()) if isinstance(data, dict) else 'N/A'}."
            )

        raw_formula = data["ltl_nl_formula"]
        self._raw_formula = raw_formula.strip()
        if not isinstance(raw_formula, str) or not raw_formula.strip():
            raise ValueError("'ltl_nl_formula' must be a non-empty string.")

        self.pi_map = _normalize_pi_predicates(data["pi_predicates"])
        if not self.pi_map:
            raise ValueError(
                "LLM returned no valid predicates (pi_predicates empty or not parseable). "
                "Use valid robot instructions."
            )
        # Tasks are unique by pi key (pi_1, pi_2, ...); descriptions may repeat (e.g. "turn 90 degrees" three times).
        self._last_returned_predicate_key: Optional[str] = None

        print(f"[LTL Planner] Generated Formula: {raw_formula}")
        print(f"[LTL Planner] Predicates: {self.pi_map}")

        spot_formula = raw_formula.replace("pi_", "p")
        try:
            self.automaton = spot.translate(spot_formula, "monitor", "det")
        except Exception as e:
            raise ValueError(
                f"Spot could not translate LTL formula '{spot_formula}': {e}. "
                "Formula may be invalid or use unsupported operators."
            ) from e
        self._add_sink_state()
        self.current_automaton_state = self.automaton.get_init_state_number()
        self.finished = False
        self.constraint_predicates = self._classify_predicates()
        if self.constraint_predicates:
            print(f"[LTL Planner] Constraints: {self.constraint_predicates}")

    def _add_sink_state(self) -> None:
        """
        Add a sink state and connect any state that has no outgoing edge to a
        different state (e.g. the last instruction's accepting state) to the
        sink. The edge condition is the BDD for the *last* predicate in pi_map
        order, so get_next_predicate() returns that predicate (not an earlier
        one) when in such a state.
        """
        self._sink_state = None
        if not self.pi_map or self.automaton is None:
            return
        try:
            n = self.automaton.num_states()
            self.automaton.new_states(1)
            self._sink_state = n
            last_key = list(self.pi_map.keys())[-1]
            last_p_idx = _predicate_key_to_index(last_key)
            bdd_sink_cond = self._get_bdd_for_single_task(last_p_idx)
            for s in range(n):
                has_outgoing_to_other = any(
                    edge.dst != s for edge in self.automaton.out(s)
                )
                if not has_outgoing_to_other:
                    self.automaton.new_edge(s, self._sink_state, bdd_sink_cond)
        except (ValueError, RuntimeError, AttributeError) as e:
            print(f"[LTL Planner] Could not add sink state: {e}. Continuing without sink.")
            self._sink_state = None

    def _get_bdd_for_single_task(self, active_p_idx: int):
        """
        Constructs a BDD representing the state:
        "p{active_p_idx} is TRUE, and all other known p's are FALSE".
        """
        if not self.pi_map or self.automaton is None:
            raise ValueError("Cannot build BDD: no predicates or automaton.")
        clauses = []
        for key in self.pi_map.keys():
            idx = _predicate_key_to_index(key)
            if idx == active_p_idx:
                clauses.append(f"p{idx}")
            else:
                clauses.append(f"!p{idx}")
        formula_str = " & ".join(clauses)
        f = spot.formula(formula_str)
        return spot.formula_to_bdd(f, self.automaton.get_dict(), self.automaton)

    def get_next_predicate(self) -> Optional[str]:
        """
        Finds the next task by testing which 'p' allows us to leave the current state.
        """
        if self.finished:
            return None
        if not self.pi_map or self.automaton is None:
            return None
        if self._sink_state is not None and self.current_automaton_state == self._sink_state:
            self.finished = True
            return None

        bdd_false = spot.formula_to_bdd(
            spot.formula("0"), self.automaton.get_dict(), self.automaton
        )

        for key in self.pi_map.keys():
            if key in self.constraint_predicates:
                continue
            p_idx = _predicate_key_to_index(key)
            try:
                test_world_bdd = self._get_bdd_for_single_task(p_idx)
            except ValueError:
                continue
            for edge in self.automaton.out(self.current_automaton_state):
                if edge.dst == self.current_automaton_state:
                    continue
                if edge.dst == self._sink_state:
                    continue
                if (test_world_bdd & edge.cond) != bdd_false:
                    self._last_returned_predicate_key = key
                    return self.pi_map[key]

        # No goal predicate triggers a forward (non-self-loop, non-sink) edge.
        print("[LTL Planner] No tasks trigger a state change. Mission Complete.")
        self.finished = True
        return None

    def advance_state(self, finished_task_nl: str) -> None:
        """Updates automaton state when a subgoal is confirmed. Uses the predicate last returned by get_next_predicate() so duplicate descriptions (e.g. 'turn 90 degrees' three times) are supported."""
        if not self.pi_map or self.automaton is None:
            return
        pi_key = self._last_returned_predicate_key
        if pi_key is None:
            print("[LTL Planner] Warning: no current predicate key (get_next_predicate not called or returned None).")
            return
        p_idx = _predicate_key_to_index(pi_key)
        try:
            current_world_bdd = self._get_bdd_for_single_task(p_idx)
        except ValueError:
            return
        bdd_false = spot.formula_to_bdd(
            spot.formula("0"), self.automaton.get_dict(), self.automaton
        )

        found_next = False
        for edge in self.automaton.out(self.current_automaton_state):
            if edge.dst == self.current_automaton_state:
                continue
            if (current_world_bdd & edge.cond) != bdd_false:
                print(f"[LTL Planner] Task '{pi_key}' satisfied edge condition.")
                print(
                    f"[LTL Planner] Transitioning State: {self.current_automaton_state} -> {edge.dst}"
                )
                self.current_automaton_state = edge.dst
                found_next = True
                break

        if not found_next:
            # No matching edge: move to sink if present (then get_next_predicate returns None),
            # otherwise mark mission complete to avoid invalid state numbers.
            if self._sink_state is not None:
                self.current_automaton_state = self._sink_state
                print(
                    f"[LTL Planner] Task '{finished_task_nl}' completed. "
                    "Transitioning to sink (mission complete)."
                )
            else:
                self.finished = True
                print(
                    f"[LTL Planner] Task '{finished_task_nl}' completed but no outgoing edge; "
                    "marking mission complete."
                )

    def _classify_predicates(self) -> dict[str, str]:
        """Classify predicates as constraints vs goals using automaton structure.

        A predicate is a CONSTRAINT (not a goal) if it never has a
        forward-progressing edge (dst != src) at ANY state in the automaton.
        Edges to the sink state (added by _add_sink_state) are excluded since
        the sink is an implementation artifact, not a genuine LTL transition.
        """
        if not self.pi_map or self.automaton is None:
            return {}

        bdd_false = spot.formula_to_bdd(
            spot.formula("0"), self.automaton.get_dict(), self.automaton
        )
        num_states = self.automaton.num_states()
        constraints: dict[str, str] = {}

        for key in self.pi_map:
            p_idx = _predicate_key_to_index(key)
            try:
                test_bdd = self._get_bdd_for_single_task(p_idx)
            except ValueError:
                continue
            is_goal = False

            for state in range(num_states):
                if state == self._sink_state:
                    continue
                for edge in self.automaton.out(state):
                    if edge.dst == state:
                        continue
                    if edge.dst == self._sink_state:
                        continue
                    if (test_bdd & edge.cond) != bdd_false:
                        is_goal = True
                        break
                if is_goal:
                    break

            if not is_goal:
                constraints[key] = self.pi_map[key]

        return constraints

    def get_active_constraints(self) -> list[str]:
        """Return natural-language descriptions of constraints active at the current state.

        An active constraint at the current state is one where making the
        predicate true produces NO valid edge at all (not even a self-loop).
        """
        if self.finished or not self.constraint_predicates or self.automaton is None:
            return []

        bdd_false = spot.formula_to_bdd(
            spot.formula("0"), self.automaton.get_dict(), self.automaton
        )
        active: list[str] = []

        for key, nl_desc in self.constraint_predicates.items():
            p_idx = _predicate_key_to_index(key)
            try:
                test_bdd = self._get_bdd_for_single_task(p_idx)
            except ValueError:
                continue

            has_any_edge = False
            for edge in self.automaton.out(self.current_automaton_state):
                if (test_bdd & edge.cond) != bdd_false:
                    has_any_edge = True
                    break

            if not has_any_edge:
                active.append(nl_desc)

        return active

