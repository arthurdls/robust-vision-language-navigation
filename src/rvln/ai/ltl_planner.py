"""
LTL Symbolic Planner: parses natural language to LTL-NL via LLM, then uses Spot
to manage the automaton state and determine the next short-horizon subgoal.
"""

from dataclasses import dataclass
from typing import Literal, Optional

try:
    import spot
except ImportError:
    spot = None

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

    def __init__(
        self,
        llm_interface: LLMUserInterface,
        use_constraints: bool = True,
    ):
        if spot is None:
            raise ImportError(
                "The 'spot' library is required for LTLSymbolicPlanner. "
                "Install via the rvln-sim conda environment."
            )
        self.llm_interface = llm_interface
        self.use_constraints = use_constraints
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
        # Classify before adding the sink state (ordering is harmless since
        # formula-structural classification doesn't inspect the automaton).
        if self.use_constraints:
            self.constraint_predicates = self._classify_predicates()
        else:
            self.constraint_predicates = {}
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

    def _active_positive_constraint_indices(self, state: Optional[int] = None) -> set[int]:
        """Indices of positive constraints that are still enforced at ``state``.

        Scoped positive constraints (``p U q``) are released once their
        right-hand side q has fired: from that point on the automaton has
        outgoing edges that allow p to be FALSE. Forcing every positive
        predicate TRUE in goal-check BDDs would incorrectly exclude those
        edges and cause goal advancement to fail.

        A positive constraint at index i is "active" at state s if making
        p_i FALSE (with all OTHER positive predicates TRUE) leaves no valid
        outgoing edge from s. If at least one edge accepts p_i = FALSE, the
        constraint has been released at this state.
        """
        if state is None:
            state = self.current_automaton_state
        if not self.constraint_predicates or self.automaton is None:
            return set()
        all_positive = self._positive_constraint_indices()
        active: set[int] = set()
        for idx in all_positive:
            try:
                violation_bdd = self._build_bdd(all_positive - {idx})
            except ValueError:
                continue
            has_any_edge = any(
                (violation_bdd & edge.cond) != self._bdd_false
                for edge in self.automaton.out(state)
            )
            if not has_any_edge:
                active.add(idx)
        return active

    def _get_bdd_for_single_task(self, active_p_idx: int):
        """BDD: only the given predicate TRUE, everything else FALSE."""
        return self._build_bdd({active_p_idx})

    def _get_bdd_goal_check(
        self, goal_p_idx: int, state: Optional[int] = None,
    ):
        """BDD: goal predicate TRUE, ACTIVE positive constraints TRUE, rest FALSE.

        Uses ``_active_positive_constraint_indices(state)`` rather than every
        statically-classified positive constraint, so that goal advancement
        works correctly past the release point of a scoped maintenance
        constraint (``p U q``).
        """
        return self._build_bdd(
            {goal_p_idx} | self._active_positive_constraint_indices(state)
        )

    def _get_bdd_constraint_violation(self, key: str):
        """BDD representing the violation state for a constraint.

        Negative: predicate TRUE (violation), every other positive constraint
        TRUE (so they don't mask the check), everything else FALSE.
        Positive: predicate FALSE (violation), every other positive constraint
        TRUE (satisfied), everything else FALSE.

        Uses the static set of positive constraints, not the per-state active
        set, to match the probe semantics in
        ``_active_positive_constraint_indices`` and avoid recursive lookups.
        """
        info = self.constraint_predicates[key]
        target_idx = _predicate_key_to_index(key)
        positive = self._positive_constraint_indices()

        if info.polarity == "negative":
            return self._build_bdd({target_idx} | positive)
        else:
            return self._build_bdd(positive - {target_idx})

    # ------------------------------------------------------------------
    # Automaton management
    # ------------------------------------------------------------------

    def _add_sink_state(self) -> None:
        """Add a sink state and connect dead-end states to it.

        States with no outgoing edge to a different state are connected to
        the sink. The edge condition uses the last *goal* predicate's
        goal-check BDD (which includes positive constraints) so that
        get_next_predicate's sink fallback can match goal completions even
        when the last key in pi_map is a constraint rather than a goal.
        """
        self._sink_state = None
        if not self.pi_map or self.automaton is None:
            return
        try:
            goal_keys = [k for k in self.pi_map if k not in self.constraint_predicates]
            if not goal_keys:
                return
            n = self.automaton.num_states()
            self.automaton.new_states(1)
            self._sink_state = n
            last_goal_key = goal_keys[-1]
            last_goal_idx = _predicate_key_to_index(last_goal_key)
            bdd_sink_cond = self._get_bdd_goal_check(last_goal_idx)
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
        """Classify predicates as constraints vs goals by walking the formula tree.

        Deterministic rules (no automaton probing):
          G(pN)              -> positive constraint (maintain)
          G(!pN)             -> negative constraint (avoid)
          F(...)             -> all APs underneath are goals
          positive pN left of U -> positive constraint (maintain until right side)
          negated !pN left of U where pN is a goal -> sequencing (not a constraint)
          negated !pN left of U where pN is NOT a goal -> negative scoped constraint
          right of U         -> goal
          bare AP / default  -> goal
        """
        if not self.pi_map or self.automaton is None:
            return {}

        spot_str = self._raw_formula.replace("pi_", "p")
        try:
            tree = spot.formula(spot_str)
        except Exception:
            return {}

        goal_aps: set[str] = set()
        self._collect_goal_aps(tree, goal_aps)

        ap_constraints: dict[str, ConstraintInfo] = {}
        self._walk_classify(tree, ap_constraints, goal_aps)

        result: dict[str, ConstraintInfo] = {}
        for ap_name, info in ap_constraints.items():
            if ap_name.startswith("p") and ap_name[1:].isdigit():
                pi_key = f"pi_{ap_name[1:]}"
                if pi_key in self.pi_map:
                    result[pi_key] = info
        return result

    def _collect_goal_aps(self, node, goals: set[str]) -> None:
        """Collect APs that are goals (under F, or on the right side of U)."""
        kind = node.kind()
        if kind == spot.op_F:
            self._collect_all_aps(node[0], goals)
            return
        if kind == spot.op_U:
            self._collect_all_aps(node[1], goals)
            self._collect_goal_aps(node[0], goals)
            return
        for i in range(node.size()):
            self._collect_goal_aps(node[i], goals)

    def _collect_all_aps(self, node, aps: set[str]) -> None:
        """Collect every AP referenced under a subtree."""
        if node.kind() == spot.op_ap:
            aps.add(str(node))
            return
        for i in range(node.size()):
            self._collect_all_aps(node[i], aps)

    def _walk_classify(self, node, out: dict[str, ConstraintInfo],
                       goal_aps: set[str]) -> None:
        kind = node.kind()
        if kind == spot.op_G:
            self._classify_under_g(node[0], out)
            return
        if kind == spot.op_F:
            return
        if kind == spot.op_U:
            self._classify_until_left(node[0], out, goal_aps)
            return
        for i in range(node.size()):
            self._walk_classify(node[i], out, goal_aps)

    def _classify_under_g(self, node, out: dict[str, ConstraintInfo]) -> None:
        kind = node.kind()
        if kind == spot.op_ap:
            ap = str(node)
            out[ap] = ConstraintInfo(
                description=self._ap_description(ap), polarity="positive",
            )
            return
        if kind == spot.op_Not and node[0].kind() == spot.op_ap:
            ap = str(node[0])
            out[ap] = ConstraintInfo(
                description=self._ap_description(ap), polarity="negative",
            )
            return
        if kind in (spot.op_F, spot.op_U, spot.op_G):
            self._walk_classify(node, out, set())
            return
        # Disjunction / implication / xor under G describe a *compound*
        # maintenance condition (e.g., G(p1 | p2) means "at all times at
        # least one of p1, p2 holds"). Splitting into independent positive
        # constraints would be strictly stronger than the formula and
        # would force every disjunct TRUE in goal-check BDDs, blocking
        # legitimate forward edges. The Spot automaton already encodes
        # the disjunction in its edge conditions; do not duplicate the
        # constraint in our prompt-side classification, and warn so we
        # notice if the LLM ever generates such a form.
        if kind in (spot.op_Or, spot.op_Implies, spot.op_Xor):
            print(
                "[LTL Planner] Warning: compound maintenance constraint "
                f"under G ({node}) is not decomposed into per-AP positive "
                "constraints. Enforcement relies on the Spot automaton "
                "edge conditions only; the goal-adherence monitor will not "
                "see these APs as MAINTAIN items in its prompt."
            )
            return
        # Conjunction or any other operator: recurse so each conjunct can
        # be classified independently (G(p1 & p2) -> two positive constraints).
        for i in range(node.size()):
            self._classify_under_g(node[i], out)

    def _classify_until_left(self, node, out: dict[str, ConstraintInfo],
                             goal_aps: set[str]) -> None:
        kind = node.kind()
        if kind == spot.op_ap:
            ap = str(node)
            out[ap] = ConstraintInfo(
                description=self._ap_description(ap), polarity="positive",
            )
            return
        if kind == spot.op_Not and node[0].kind() == spot.op_ap:
            ap = str(node[0])
            if ap not in goal_aps:
                out[ap] = ConstraintInfo(
                    description=self._ap_description(ap), polarity="negative",
                )
            return
        for i in range(node.size()):
            self._classify_until_left(node[i], out, goal_aps)

    def _ap_description(self, ap_name: str) -> str:
        if ap_name.startswith("p") and ap_name[1:].isdigit():
            return self.pi_map.get(f"pi_{ap_name[1:]}", ap_name)
        return ap_name

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
