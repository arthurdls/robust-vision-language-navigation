"""
Notes:
We use Co-safe LTL forumlas (formulas that exclude always so that they can be solved in finite horizon time)

LTL-formula Spot installation: https://spot.lre.epita.fr/install.html
conda install -c conda-forge spot


Tasks to complete:
    Break down into action and object for each predicate
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    import spot
except ImportError:
    spot = None
from ..config import DEFAULT_LLM_MODEL
from ..paths import FORMULA_CACHE_DIR
from .prompts import (
    LTL_NL_SYSTEM_PROMPT,
    LTL_NL_EXAMPLES_PROMPT,
    LTL_NL_SYSTEM_PROMPT_SEQUENTIAL,
    LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL,
    LTL_NL_RESTATED_TASK_PROMPT,
    LTL_NL_CHECK_PREDICATES_PROMPT,
    LTL_NL_CHECK_SEMANTICS_PROMPT,
)
from .utils.llm_providers import LLMFactory
from .utils.parsing import parse_ltl_nl, extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formula cache
# ---------------------------------------------------------------------------
#
# The LTL planner is re-instantiated for every episode, but the same NL
# instruction produces the same formula deterministically only when the LLM
# is well-behaved. To keep the formula stable across episodes (especially
# across the 3 starting-position variants of each task), we cache the
# planner LLM output on disk under cached_formulas/{hash}.json.
#
# The cache key includes the model name and the prompt-text version (a
# digest of the system prompts) so that prompt revisions invalidate stale
# entries automatically. The cache is committed to git for reproducibility.
#
# Disable via environment variable RVLN_IGNORE_FORMULA_CACHE=1 or by passing
# ``ignore_cache=True`` to ``make_natural_language_request``.

def _prompt_version_for(prompts: tuple) -> str:
    h = hashlib.sha1()
    for prompt in prompts:
        h.update(prompt.encode("utf-8"))
    return h.hexdigest()[:12]


def _formula_cache_key(instruction: str, model: str, prompt_version: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt_version": prompt_version,
        "instruction": instruction.strip(),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _formula_cache_path(instruction: str, model: str, prompt_version: str) -> Path:
    return FORMULA_CACHE_DIR / f"{_formula_cache_key(instruction, model, prompt_version)}.json"


def _load_cached_formula(
    instruction: str, model: str, prompt_version: str,
) -> Optional[Dict[str, Any]]:
    if os.environ.get("RVLN_IGNORE_FORMULA_CACHE") == "1":
        return None
    path = _formula_cache_path(instruction, model, prompt_version)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            entry = json.load(f)
    except Exception as e:
        logger.warning("Failed to read formula cache %s: %s", path, e)
        return None
    if not isinstance(entry, dict):
        return None
    if "ltl_nl_formula" not in entry or "pi_predicates" not in entry:
        return None
    return entry


def _save_cached_formula(
    instruction: str, model: str, prompt_version: str,
    formula: Dict[str, Any], raw_response: str,
) -> None:
    FORMULA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _formula_cache_path(instruction, model, prompt_version)
    payload = {
        "model": model,
        "prompt_version": prompt_version,
        "instruction": instruction.strip(),
        "ltl_nl_formula": formula.get("ltl_nl_formula", ""),
        "pi_predicates": formula.get("pi_predicates", {}),
        "raw_response": raw_response,
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Failed to write formula cache %s: %s", path, e)

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class LLMUserInterface():
    """Represents the LLM-User interface used to communicate with the robot.

    Attributes:
        current_ltl_nl_formula: The LTL-NL Formula the robot is currently working with.
    Examples:
    >>> interface = LLMUserInterface()
    >>> command = "Eventually deliver the pen to location D, but only after you have delivered either a drink or an apple to location E."
    >>> interface.make_natural_language_request(command)
    >>> interface.ltl_nl_formula
    {
      "pi_predicates": {
          "pi_1": "Deliver pen to Location D",
          "pi_2": "Deliver drink to Location E",
          "pi_3": "Deliver apple to Location E"
      },
      "ltl_nl_formula": "F pi_1 & (!pi_1 U (pi_2 | pi_3))"
    }
    """

    def __init__(self, model: str = DEFAULT_LLM_MODEL, use_constraints: bool = True):
        self._model = model
        self._use_constraints = use_constraints
        self._base_llm = LLMFactory.create(model=model, rate_limit_seconds=0.0)

        if use_constraints:
            system_prompt = LTL_NL_SYSTEM_PROMPT
            examples_prompt = LTL_NL_EXAMPLES_PROMPT
        else:
            system_prompt = LTL_NL_SYSTEM_PROMPT_SEQUENTIAL
            examples_prompt = LTL_NL_EXAMPLES_PROMPT_SEQUENTIAL

        self._initial_context = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": examples_prompt},
            {"role": "system", "content": LTL_NL_RESTATED_TASK_PROMPT},
        ]
        self._prompt_version = _prompt_version_for(
            (system_prompt, examples_prompt, LTL_NL_RESTATED_TASK_PROMPT)
        )

        self._history = list(self._initial_context)
        self.ltl_nl_formula = {}
        self._ltl_is_confirmed = False
        self.llm_call_records: List[Dict[str, Any]] = []

    def make_natural_language_request(
        self, request: str, ignore_cache: bool = False,
    ) -> str:
        """Makes a natural language request to the robot.

        Args:
            request: The natural language request the robot should satisfy
            ignore_cache: If True, skip the on-disk formula cache and always
                call the LLM. Default False reuses any cached formula for
                this (model, prompt-version, instruction) tuple.

        Returns:
            The LLM response to the natural language request. If the request is valid,
            the LLM response will check to confirm that the request was interpreted correctly.
            However, if the request was invalid, the LLM response will ask the user to input
            a valid response.
        """
        self.reset_to_baseline_context()

        cached = None if ignore_cache else _load_cached_formula(
            request, self._model, self._prompt_version,
        )
        if cached is not None:
            self.ltl_nl_formula = {
                "ltl_nl_formula": cached["ltl_nl_formula"],
                "pi_predicates": cached["pi_predicates"],
            }
            self.llm_call_records.append({
                "label": "ltl_nl_planning_cached",
                "rtt_s": 0.0,
                "model": cached.get("model", self._model),
                "input_tokens": 0,
                "output_tokens": 0,
            })
            response_text = cached.get("raw_response", "")
            self._history.append({"role": "user", "content": request})
            self._history.append({"role": "assistant", "content": response_text})
            return self.ltl_nl_to_string()

        self._history.append({
                "role": "user",
                "content": request,
            })
        t0 = time.time()
        response_text = self._base_llm.make_request(self._history, temperature=0.0, json_mode=True)
        rtt = time.time() - t0
        usage = self._base_llm.last_usage
        self.llm_call_records.append({
            "label": "ltl_nl_planning",
            "rtt_s": round(rtt, 3),
            "model": usage.get("model", self._model),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })

        try:
            ltl_nl = extract_json(response_text)
        except (json.JSONDecodeError, SyntaxError) as e:
            print(f"Error decoding JSON: {e}")
            raise Exception("INCOMPLETE: Unable to find ltl-nl formula")

        self.ltl_nl_formula = ltl_nl
        self._history.append({
                "role": "assistant",
                "content": response_text,
            })

        if not ignore_cache and isinstance(ltl_nl, dict):
            _save_cached_formula(
                request, self._model, self._prompt_version, ltl_nl, response_text,
            )

        return self.ltl_nl_to_string()

    def ltl_nl_to_string(self) -> str:
        if self.ltl_nl_formula != {}:
            formula_string, predicates = self.ltl_nl_formula['ltl_nl_formula'], self.ltl_nl_formula['pi_predicates']
            return parse_ltl_nl(formula_string, predicates)
        else:
            return ""

    def reset_to_baseline_context(self) -> None:
        """Resets model back to initial context.
        """
        self._history = list(self._initial_context)

    def evaluate_interface(self, dataset_path: str, reset_after_each_command=False) -> None:
        """Evaluates LLM-User Interface on a natural language to LTL-NL convertion database.

        Args:
            dataset_path: The location of the dataset which takes the form: "'natural language command'?{json ltl-nl};" for every test case.
        Returns:
            summary on dataset evaluation
        """
        evaluation_llm = self._base_llm
        def check_predicates(predicates_1: dict, predicates_2: dict) -> bool:
            if len(predicates_1) != len(predicates_2):
                return False
            sorted_predicates_1 = sorted(predicates_1.items())
            sorted_predicates_2 = sorted(predicates_2.items())
            request = LTL_NL_CHECK_PREDICATES_PROMPT

            for p1, p2 in zip(sorted_predicates_1, sorted_predicates_2):
                p1_text = p1[-1]
                p2_text = p2[-1]
                request += f"Pair:\nStatement 1: '{p1_text}'\nStatement 2: '{p2_text}'\n\n"

            response_text = evaluation_llm.make_text_request(request, temperature=0.0)
            left_curly_index = response_text.rfind("{")
            right_curly_index = response_text.rfind("}")
            response_text = response_text[left_curly_index + 1: right_curly_index]
            if int(response_text) == 1:
                return True
            elif int(response_text) == 0:
                return False
            else:
                raise ValueError(f"Failed to check predicates. Response is '{response_text}'")

        def check_semantics(ltl_nl_str_1: str, ltl_nl_str_2: str) -> bool:
            request = LTL_NL_CHECK_SEMANTICS_PROMPT.format(
                ltl_nl_str_1=ltl_nl_str_1,
                ltl_nl_str_2=ltl_nl_str_2,
            )

            response_text = evaluation_llm.make_text_request(request, temperature=0.0).strip()
            try:
                value = int(response_text)
            except ValueError:
                raise ValueError(f"Failed to check semantics. Response is '{response_text}'")
            if value == 1:
                return True
            elif value == 0:
                return False
            else:
                raise ValueError(f"Failed to check semantics. Response is '{response_text}'")

        with open(dataset_path, "r") as f:
            data = "".join(f.readlines())

            # Because of the nature of the dataset, remove the last element after splitting with ';'
            data = data.replace("\n","").split(";")[:-1]
            counter = 0
            total_correct_formulas = 0
            total_correct_predicates = 0
            total_correct_semantics = 0
            num_datapoints = len(data)
            for datapoint in data:
                counter += 1
                nl_command, formula_spec = datapoint.split("?")
                expected_formula_spec = json.loads(formula_spec)
                try:
                    self.make_natural_language_request(nl_command)
                except Exception as e:
                    print(f"ERROR making nl request: {e}")
                    continue

                pred_formula_spec = self.ltl_nl_formula
                if "null" in dataset_path:
                    if pred_formula_spec == expected_formula_spec:
                        total_correct_formulas += 1
                        total_correct_predicates += 1
                        total_correct_semantics += 1
                        print(f"{counter} correct predicates")
                        print(f"{counter} correct formulas")
                        print(f"{counter} correct semantics")
                    continue

                expected_formula: str = expected_formula_spec['ltl_nl_formula']
                pred_formula: str = pred_formula_spec['ltl_nl_formula']
                expected_predicates: dict = expected_formula_spec['pi_predicates']
                pred_predicates: dict = pred_formula_spec['pi_predicates']
                try:
                    equal_predicates = check_predicates(pred_predicates, expected_predicates)
                    equal_semantics = check_semantics(nl_command, self.ltl_nl_to_string())
                except Exception as e:
                    print(f"ERROR checking either predicates or semantics: {e}")
                    continue

                if equal_predicates:
                    print(f"{counter} correct predicates")
                    total_correct_predicates += 1
                    if spot.are_equivalent(pred_formula, expected_formula):
                        print(f"{counter} correct formulas")
                        total_correct_formulas += 1
                    else:
                        print(f"{counter} Expected formula: {expected_formula}")
                        print(f"{counter} Predicted formula: {pred_formula}")
                else:
                    print(f"{counter} Expected predicates: {expected_predicates}")
                    print(f"{counter} Predicted predicates: {pred_predicates}")

                if equal_semantics:
                    print(f"{counter} correct semantics")
                    total_correct_semantics += 1

                if reset_after_each_command:
                    self.reset_to_baseline_context()
        out = []
        out += [f"Eval on {dataset_path} pi_predicates (out of total datapoints): {total_correct_predicates}/{num_datapoints}"]
        out += [f"Eval on {dataset_path} ltl_nl_formula (out of correct predicates): {total_correct_formulas}/{total_correct_predicates}"]
        out += [f"Eval on {dataset_path} semantics (out of total datapoints): {total_correct_semantics}/{num_datapoints}"]
        out = "\n".join(out) + "\n"
        print(out)
        return out




def run_REPL():
    interface = LLMUserInterface()

    while True:
        user_input = input("user:~$ ")
        if user_input == "quit" or user_input == "exit":
            break
        if user_input == "reset":
            interface.reset_to_baseline_context()
            continue
        if user_input == "manual mode":
            print("NOT YET IMPLEMENTED")
            continue
        if user_input == "help":
            print("Supported commands: quit, exit, reset")
            continue
        if user_input == "print ltl-nl":
            print(interface.ltl_nl_formula)
            continue
        response = interface.make_natural_language_request(user_input)
        print(f"interface: {response}")


def run_test_data():
    interface = LLMUserInterface()
    eval_out = []
    try:
        eval_out += [interface.evaluate_interface("src/data/test_commands_easy.txt")]
    except Exception as e:
        print(f"ERROR on easy: {e}")

    try:
        eval_out += [interface.evaluate_interface("src/data/test_commands_medium.txt")]
    except Exception as e:
        print(f"ERROR on medium: {e}")

    try:
        eval_out += [interface.evaluate_interface("src/data/test_commands_hard.txt")]
    except Exception as e:
        print(f"ERROR on hard: {e}")

    # try:
    #     eval_out += [interface.evaluate_interface("src/data/test_commands_null.txt")]
    # except Exception as e:
    #     print(f"ERROR on null: {e}")

    print("\n\n***Evaluation Summary***\n")
    for ev in eval_out:
        print(ev)


if __name__ == "__main__":
    # run_test_data()
    run_REPL()