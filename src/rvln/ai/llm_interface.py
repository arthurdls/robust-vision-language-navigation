"""
Notes:
We use Co-safe LTL forumlas (formulas that exclude always so that they can be solved in finite horizon time)

LTL-formula Spot installation: https://spot.lre.epita.fr/install.html
conda install -c conda-forge spot


Tasks to complete:
    Break down into action and object for each predicate
"""
import json
import logging
import time
from typing import Any, Dict, List
import spot
from ..config import DEFAULT_LLM_MODEL
from .prompts import (
    LTL_NL_SYSTEM_PROMPT,
    LTL_NL_EXAMPLES_PROMPT,
    LTL_NL_RESTATED_TASK_PROMPT,
    LTL_NL_CHECK_PREDICATES_PROMPT,
    LTL_NL_CHECK_SEMANTICS_PROMPT,
)
from .utils.llm_providers import LLMFactory
from .utils.parsing import parse_ltl_nl, extract_json

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

    def __init__(self, model: str = DEFAULT_LLM_MODEL):
        self._model = model
        self._base_llm = LLMFactory.create(model=model, rate_limit_seconds=0.0)

        self._initial_context = [
            {"role": "system", "content": LTL_NL_SYSTEM_PROMPT},
            {"role": "system", "content": LTL_NL_EXAMPLES_PROMPT},
            {"role": "system", "content": LTL_NL_RESTATED_TASK_PROMPT},
        ]

        self._history = list(self._initial_context)
        self.ltl_nl_formula = {}
        self._ltl_is_confirmed = False
        self.llm_call_records: List[Dict[str, Any]] = []

    def make_natural_language_request(self, request: str) -> str:
        """Makes a natural language request to the robot.

        Args:
            request: The natural language request the robot should satisfy

        Returns:
            The LLM response to the natural language request. If the request is valid,
            the LLM response will check to confirm that the request was interpreted correctly.
            However, if the request was invalid, the LLM response will ask the user to input
            a valid response.
        """
        self.reset_to_baseline_context()

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
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            raise Exception("INCOMPLETE: Unable to find ltl-nl formula")

        self.ltl_nl_formula = ltl_nl
        self._history.append({
                "role": "assistant",
                "content": response_text,
            })
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

            response_text = evaluation_llm.make_text_request(request, temperature=0.0)
            if int(response_text) == 1:
                return True
            elif int(response_text) == 0:
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