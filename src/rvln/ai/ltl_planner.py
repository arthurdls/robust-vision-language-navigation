"""
LTL Symbolic Planner: parses natural language to LTL-NL via LLM, then uses Spot
to manage the automaton state and determine the next short-horizon subgoal.
"""

import re
from dataclasses import dataclass
from typing import Literal, Optional

import spot

from .llm_interface import LLMUserInterface


@dataclass
class ConstraintInfo:
    description: str
    polarity: Literal["negative", "positive"]


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
        self._sink_state: Optional[int] = None
        self.pi_map = {}
        self._last_returned_predicate_key: Optional[str] = None
        self.finished = False
        self._raw_formula: str = ""
        self.constraint_predicates: dict[str, ConstraintInfo] = {}
        self._bdd_false = None

    def plan_from_natural_language(self, instruction: str) -> None:
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
        self._last_returned_predicate_key = None

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

        self._bdd_false = spot.formula_to_bdd(
            spot.formula("0"), self.automaton.get_dict(), self.automaton
        )
        # IMPORTANT: classify predicates BEFORE adding the sink state.
        # _classify_predicates uses _has_forward_edge which skips sink edges,
        # so if the sink exists during classification, the last goal predicate
        # (whose only forward edge leads to the sink) gets misclassified as a
        # constraint and is never returned by get_next_predicate.
        self.constraint_predicates = self._classify_predicates()
        self._add_sink_state()
        self.current_automaton_state = self.automaton.get_init_state_number()
        self.finished = False
        if self.constraint_predicates:
            print(f"[LTL Planner] Constraints: {self.constraint_predicates}")

    # ------------------------------------------------------------------
    # BDD construction
    # ------------------------------------------------------------------

    def _build_bdd(self, true_indices: set[int]):
        """Build a BDD with the given predicate indices TRUE, all others FALSE."""
        if not self.pi_map or self.automaton is None:
            raise ValueError("Cannot build BDD: no predicates or automaton.")
        clauses = []
        for key in self.pi_map:
            idx = _predicate_key_to_index(key)
            clauses.append(f"p{idx}" if idx in true_indices else f"!p{idx}")
        return spot.formula_to_bdd(
            spot.formula(" & ".join(clauses)),
            self.automaton.get_dict(),
            self.automaton,
        )

    def _positive_constraint_indices(self) -> set[int]:
        return {
            _predicate_key_to_index(k)
            for k, info in self.constraint_predicates.items()
            if info.polarity == "positive"
        }

    def _get_bdd_for_single_task(self, active_p_idx: int):
        """BDD: only the given predicate TRUE, everything else FALSE."""
        return self._build_bdd({active_p_idx})

    def _get_bdd_goal_check(self, goal_p_idx: int):
        """BDD: goal predicate TRUE, positive constraints TRUE, rest FALSE."""
        return self._build_bdd({goal_p_idx} | self._positive_constraint_indices())

    def _get_bdd_all_false(self):
        """BDD: ALL predicates FALSE."""
        return self._build_bdd(set())

    def _get_bdd_constraint_violation(self, key: str):
        """BDD representing the violation state for a constraint.

        Negative: predicate TRUE (violation), positive constraints TRUE
        (satisfied so they don't mask the check), everything else FALSE.
        Positive: predicate FALSE (violation), other positive constraints
        TRUE (satisfied), everything else FALSE.
        """
        info = self.constraint_predicates[key]
        target_idx = _predicate_key_to_index(key)
        positive = self._positive_constraint_indices()

        if info.polarity == "negative":
            return self._build_bdd({target_idx} | positive)
        else:
            return self._build_bdd((positive - {target_idx}))

    # ------------------------------------------------------------------
    # Automaton management
    # ------------------------------------------------------------------

    def _add_sink_state(self) -> None:
        """Add a sink state and connect dead-end states to it.

        States with no outgoing edge to a different state are connected to
        the sink. The edge condition uses the last predicate's BDD so
        get_next_predicate returns that predicate when in such a state.
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

    def get_next_predicate(self) -> Optional[str]:
        """Find the next goal by testing which predicate allows leaving the current state."""
        if self.finished:
            return None
        if not self.pi_map or self.automaton is None:
            return None
        if self._sink_state is not None and self.current_automaton_state == self._sink_state:
            self.finished = True
            return None

        for key in self.pi_map:
            if key in self.constraint_predicates:
                continue
            p_idx = _predicate_key_to_index(key)
            try:
                test_world_bdd = self._get_bdd_goal_check(p_idx)
            except ValueError:
                continue
            for edge in self.automaton.out(self.current_automaton_state):
                if edge.dst == self.current_automaton_state:
                    continue
                if edge.dst == self._sink_state:
                    continue
                if (test_world_bdd & edge.cond) != self._bdd_false:
                    self._last_returned_predicate_key = key
                    return self.pi_map[key]

        # BUG FIX: on some Spot versions (or after postprocessing) the monitor
        # automaton has fewer states, so the last goal's only outgoing edge
        # leads to the sink. The loop above skips sink edges, which silently
        # drops the final goal. Fall back to checking sink edges for any
        # not-yet-returned goal predicate (skip _last_returned_predicate_key
        # to avoid re-returning an already-completed goal at a true dead-end).
        if self._sink_state is not None:
            for key in self.pi_map:
                if key in self.constraint_predicates:
                    continue
                if key == self._last_returned_predicate_key:
                    continue
                p_idx = _predicate_key_to_index(key)
                try:
                    test_world_bdd = self._get_bdd_goal_check(p_idx)
                except ValueError:
                    continue
                for edge in self.automaton.out(self.current_automaton_state):
                    if edge.dst == self.current_automaton_state:
                        continue
                    if edge.dst != self._sink_state:
                        continue
                    if (test_world_bdd & edge.cond) != self._bdd_false:
                        self._last_returned_predicate_key = key
                        return self.pi_map[key]

        print("[LTL Planner] No tasks trigger a state change. Mission Complete.")
        self.finished = True
        return None

    def advance_state(self, finished_task_nl: str) -> None:
        """Update automaton state when a subgoal is confirmed.

        Uses the predicate key from the last get_next_predicate() call so
        duplicate descriptions (e.g. 'turn 90 degrees' repeated) work.
        """
        if not self.pi_map or self.automaton is None:
            return
        pi_key = self._last_returned_predicate_key
        if pi_key is None:
            print("[LTL Planner] Warning: no current predicate key "
                  "(get_next_predicate not called or returned None).")
            return
        p_idx = _predicate_key_to_index(pi_key)
        try:
            current_world_bdd = self._get_bdd_goal_check(p_idx)
        except ValueError:
            return

        found_next = False
        for edge in self.automaton.out(self.current_automaton_state):
            if edge.dst == self.current_automaton_state:
                continue
            if (current_world_bdd & edge.cond) != self._bdd_false:
                print(f"[LTL Planner] Task '{pi_key}' satisfied edge condition.")
                print(
                    f"[LTL Planner] Transitioning State: "
                    f"{self.current_automaton_state} -> {edge.dst}"
                )
                self.current_automaton_state = edge.dst
                found_next = True
                break

        if not found_next:
            if self._sink_state is not None:
                self.current_automaton_state = self._sink_state
                print(
                    f"[LTL Planner] Task '{finished_task_nl}' completed. "
                    "Transitioning to sink (mission complete)."
                )
            else:
                self.finished = True
                print(
                    f"[LTL Planner] Task '{finished_task_nl}' completed "
                    "but no outgoing edge; marking mission complete."
                )

    # ------------------------------------------------------------------
    # Constraint classification
    # ------------------------------------------------------------------

    def _classify_predicates(self) -> dict[str, ConstraintInfo]:
        """Classify predicates as constraints vs goals using automaton structure.

        Three-pass approach:
        1. Identify candidates: predicates with no forward edge when only
           that predicate is TRUE (all others FALSE).
        2. Detect polarity of each candidate.
        3. Re-check negative candidates with positive constraints TRUE. If
           forward edges appear, the candidate is a goal blocked by
           unsatisfied positive constraints in pass 1.
        """
        if not self.pi_map or self.automaton is None:
            return {}

        num_states = self.automaton.num_states()

        candidate_keys: list[str] = []
        for key in self.pi_map:
            p_idx = _predicate_key_to_index(key)
            try:
                test_bdd = self._get_bdd_for_single_task(p_idx)
            except ValueError:
                continue
            if not self._has_forward_edge(test_bdd, num_states):
                candidate_keys.append(key)

        polarities: dict[str, Literal["negative", "positive"]] = {}
        for key in candidate_keys:
            polarities[key] = self._detect_polarity(key)

        positive_keys = [k for k, p in polarities.items() if p == "positive"]
        positive_indices = {_predicate_key_to_index(k) for k in positive_keys}

        constraints: dict[str, ConstraintInfo] = {}
        for key in candidate_keys:
            if polarities[key] == "positive":
                constraints[key] = ConstraintInfo(
                    description=self.pi_map[key], polarity="positive",
                )
                continue

            if positive_keys:
                p_idx = _predicate_key_to_index(key)
                recheck_bdd = self._build_bdd({p_idx} | positive_indices)
                if self._has_forward_edge(recheck_bdd, num_states):
                    continue

            constraints[key] = ConstraintInfo(
                description=self.pi_map[key], polarity="negative",
            )

        # IMPORTANT: formula-level override for postprocessed/merged automata.
        # spot.postprocess(aut, "monitor", "small") (or some Spot versions
        # natively) can merge the final accepting state into an earlier state.
        # This makes the last goal's accepting transition a self-loop, causing
        # _has_forward_edge to miss it and misclassify the goal as a constraint.
        # A predicate under an F (eventually) operator is semantically a goal
        # that MUST be achieved, never a constraint. Remove any such predicate
        # from the constraint set.
        for key in list(constraints.keys()):
            p_idx = _predicate_key_to_index(key)
            if re.search(rf"\bF\s+pi_{p_idx}\b", self._raw_formula) or \
               re.search(rf"\bF\s+p{p_idx}\b", self._raw_formula):
                del constraints[key]

        return constraints

    def _has_forward_edge(self, test_bdd, num_states: int) -> bool:
        """Check if test_bdd produces any forward (non-self, non-sink) edge at any state."""
        for state in range(num_states):
            if state == self._sink_state:
                continue
            for edge in self.automaton.out(state):
                if edge.dst == state:
                    continue
                if edge.dst == self._sink_state:
                    continue
                if (test_bdd & edge.cond) != self._bdd_false:
                    return True
        return False

    def _detect_polarity(self, key: str) -> Literal["negative", "positive"]:
        """Detect whether a constraint is negative (avoid) or positive (maintain).

        Tests two BDDs at the initial state:
        - bdd_true: this predicate TRUE, all others FALSE
        - bdd_all_false: all predicates FALSE

        If bdd_true has edges but bdd_all_false does not, the predicate being
        TRUE is accepted and being FALSE is rejected: positive (maintain).
        Otherwise, default to negative (avoid).
        """
        p_idx = _predicate_key_to_index(key)
        bdd_true = self._get_bdd_for_single_task(p_idx)
        bdd_all_false = self._get_bdd_all_false()

        init = self.automaton.get_init_state_number()
        has_edge_true = any(
            (bdd_true & edge.cond) != self._bdd_false
            for edge in self.automaton.out(init)
        )
        has_edge_all_false = any(
            (bdd_all_false & edge.cond) != self._bdd_false
            for edge in self.automaton.out(init)
        )

        if has_edge_true and not has_edge_all_false:
            return "positive"
        return "negative"

    def get_active_constraints(self) -> list[ConstraintInfo]:
        """Return constraints active at the current automaton state.

        A constraint is active when its violation BDD produces NO valid
        edge at the current state (the automaton would reject the violation).
        """
        if self.finished or not self.constraint_predicates or self.automaton is None:
            return []

        active: list[ConstraintInfo] = []
        for key, info in self.constraint_predicates.items():
            try:
                test_bdd = self._get_bdd_constraint_violation(key)
            except ValueError:
                continue

            has_any_edge = any(
                (test_bdd & edge.cond) != self._bdd_false
                for edge in self.automaton.out(self.current_automaton_state)
            )
            if not has_any_edge:
                active.append(info)

        return active
