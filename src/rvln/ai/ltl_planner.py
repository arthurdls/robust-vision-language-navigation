"""
LTL Symbolic Planner: parses natural language to LTL-NL via LLM, then uses Spot
to manage the automaton state and determine the next short-horizon subgoal.

Every predicate is a goal whose completion criterion is its full natural-
language description (including any "until ...", "while ...", "for N meters",
or "without ..." clauses). The Spot automaton drives sequential subgoal
advancement.
"""

from typing import Optional

try:
    import spot
except ImportError:
    spot = None

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
        if spot is None:
            raise ImportError(
                "The 'spot' library is required for LTLSymbolicPlanner. "
                "Install via the rvln-sim conda environment."
            )
        self.llm_interface = llm_interface
        self.current_automaton_state = 0
        self.automaton = None
        self._sink_state: Optional[int] = None
        self.pi_map = {}
        self._last_returned_predicate_key: Optional[str] = None
        self._returned_keys: set = set()
        self.finished = False

    def plan_from_natural_language(self, instruction: str) -> None:
        """
        1. Query LLM for LTL formula.
        2. Convert LTL string to Spot Automaton.

        If Spot rejects the LLM-generated formula (invalid syntax,
        unsupported operators), retry the LLM call once with the Spot
        error included in the prompt and the on-disk cache disabled, so
        the model can self-correct. Raises only if both attempts fail.
        """
        if not instruction or not isinstance(instruction, str):
            raise ValueError("Instruction must be a non-empty string.")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Instruction must be a non-empty string.")

        print(f"[LTL Planner] Processing instruction: '{instruction}'")
        last_error: Optional[Exception] = None
        for attempt in (1, 2):
            request_text = instruction
            if attempt == 2 and last_error is not None:
                request_text = (
                    f"{instruction}\n\n"
                    f"Previous attempt produced a formula that Spot could not "
                    f"translate: {last_error}. Regenerate the formula using "
                    "only valid LTL syntax (operators F, G, U, &, |, !, X)."
                )
            _ = self.llm_interface.make_natural_language_request(
                request_text, ignore_cache=(attempt > 1),
            )
            data = self.llm_interface.ltl_nl_formula

            if not data or not isinstance(data, dict):
                raise ValueError("LLM could not generate a valid LTL formula (empty or non-dict response).")
            if "ltl_nl_formula" not in data or "pi_predicates" not in data:
                raise ValueError(
                    "LLM response must contain 'ltl_nl_formula' and 'pi_predicates'. "
                    f"Keys received: {list(data.keys()) if isinstance(data, dict) else 'N/A'}."
                )

            raw_formula = data["ltl_nl_formula"]
            if not isinstance(raw_formula, str) or not raw_formula.strip():
                raise ValueError("'ltl_nl_formula' must be a non-empty string.")

            self.pi_map = _normalize_pi_predicates(data["pi_predicates"])
            if not self.pi_map:
                raise ValueError(
                    "LLM returned no valid predicates (pi_predicates empty or not parseable). "
                    "Use valid robot instructions."
                )
            self._last_returned_predicate_key = None
            self._returned_keys = set()

            print(f"[LTL Planner] Generated Formula: {raw_formula}")
            print(f"[LTL Planner] Predicates: {self.pi_map}")

            spot_formula = raw_formula.replace("pi_", "p")
            try:
                self.automaton = spot.translate(spot_formula, "monitor", "det")
                break
            except Exception as e:
                last_error = e
                print(
                    f"[LTL Planner] Spot translate failed on attempt {attempt}: "
                    f"{e}. Formula was: '{spot_formula}'."
                )
                if attempt >= 2:
                    raise ValueError(
                        f"Spot could not translate LTL formula '{spot_formula}' "
                        f"after retry: {e}. Formula may be invalid or use "
                        "unsupported operators."
                    ) from e

        self._add_sink_state()
        self.current_automaton_state = self.automaton.get_init_state_number()
        self.finished = False

    def _add_sink_state(self) -> None:
        """Add a sink state and connect dead-end states to it.

        The edge condition is the BDD for the *last* predicate in pi_map
        order, so get_next_predicate() returns that predicate (not an
        earlier one) when in such a state.
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
        """BDD: the given predicate TRUE, all other known predicates FALSE."""
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
        """Find the next task by testing which predicate is consistent with the
        current automaton state, in pi_map order.

        A predicate is "consistent" if its single-task BDD intersects any
        non-sink outgoing edge (state-changing or self-loop). Allowing
        self-loops is necessary because spot.postprocess (and some Spot
        versions) merge states such that intermediate goals fire only by
        keeping the automaton in a self-loop, with no state change. Tracking
        returned keys prevents re-returning an already-completed goal whose
        self-loop still accepts.
        """
        if self.finished:
            return None
        if not self.pi_map or self.automaton is None:
            return None
        if self._sink_state is not None and self.current_automaton_state == self._sink_state:
            self.finished = True
            return None
        if len(self._returned_keys) >= len(self.pi_map):
            self.finished = True
            return None

        bdd_false = spot.formula_to_bdd(
            spot.formula("0"), self.automaton.get_dict(), self.automaton
        )

        # Primary: pi_map order, first unreturned key whose single-task BDD
        # intersects any non-sink outgoing edge (including self-loops).
        for key in self.pi_map.keys():
            if key in self._returned_keys:
                continue
            p_idx = _predicate_key_to_index(key)
            try:
                test_world_bdd = self._get_bdd_for_single_task(p_idx)
            except ValueError:
                continue
            for edge in self.automaton.out(self.current_automaton_state):
                if edge.dst == self._sink_state:
                    continue
                if (test_world_bdd & edge.cond) != bdd_false:
                    self._last_returned_predicate_key = key
                    return self.pi_map[key]

        # Sink-edge fallback: the only remaining out-edge leads to sink.
        # Common when the dead-end's only path forward is the synthesized
        # sink edge for the last predicate.
        if self._sink_state is not None:
            for key in self.pi_map.keys():
                if key in self._returned_keys:
                    continue
                p_idx = _predicate_key_to_index(key)
                try:
                    test_world_bdd = self._get_bdd_for_single_task(p_idx)
                except ValueError:
                    continue
                for edge in self.automaton.out(self.current_automaton_state):
                    if edge.dst != self._sink_state:
                        continue
                    if (test_world_bdd & edge.cond) != bdd_false:
                        self._last_returned_predicate_key = key
                        return self.pi_map[key]

        # Last-resort: return any unreturned key in pi_map order so the user
        # always sees every declared subgoal even when the monitor automaton
        # offers no edge-based confirmation.
        for key in self.pi_map.keys():
            if key in self._returned_keys:
                continue
            self._last_returned_predicate_key = key
            return self.pi_map[key]

        print("[LTL Planner] No tasks trigger a state change. Mission Complete.")
        self.finished = True
        return None

    def advance_state(self, finished_task_nl: str) -> None:
        """Update automaton state when a subgoal is confirmed.

        Uses the predicate key from the last get_next_predicate() call so
        duplicate descriptions (e.g. 'turn 90 degrees' three times) work.

        When no state-changing edge fires for the completed predicate (an
        "implicit" goal accepted only via self-loop), the state is left
        unchanged so the planner can still walk through any remaining
        unreturned predicates. Only after every key has been returned does
        the planner advance to the sink / finished state.
        """
        if not self.pi_map or self.automaton is None:
            return
        pi_key = self._last_returned_predicate_key
        if pi_key is None:
            print("[LTL Planner] Warning: no current predicate key (get_next_predicate not called or returned None).")
            return

        # Validate the caller is reporting completion of the predicate we
        # actually expect. If a monitor hallucinates completion of the wrong
        # subgoal (e.g. claims pi_3 done while the automaton state expects
        # pi_1), the formal-method advantage of LTL is bypassed silently.
        # Log it loudly so post-hoc analysis can flag the run.
        expected_text = self.pi_map.get(pi_key, "")
        if (
            isinstance(finished_task_nl, str)
            and expected_text
            and finished_task_nl.strip() != expected_text.strip()
        ):
            print(
                f"[LTL Planner] WARNING: advance_state called with "
                f"finished_task_nl='{finished_task_nl}' but the current "
                f"predicate is {pi_key}='{expected_text}'. Trusting the "
                "predicate key (advance_state's NL argument is informational)."
            )

        self._returned_keys.add(pi_key)
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
            if len(self._returned_keys) >= len(self.pi_map):
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
            else:
                print(
                    f"[LTL Planner] Task '{pi_key}' completed without state change "
                    "(implicit goal); remaining goals will be fetched next."
                )
