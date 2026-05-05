"""Manual (terminal-driven) stand-ins for the OpenAI-backed components.

Used by ``run_hardware.py --no-ai``. Each class mirrors the public surface of
its real counterpart (``LLMUserInterface``, ``SequentialLTLPlanner``,
``SubgoalConverter``, ``GoalAdherenceMonitor``) but reads decisions from stdin
instead of calling an LLM/VLM. Intended for hardware bring-up, debugging the
control loop, or running without an OPENAI_API_KEY.

Limitations:
 - Only the synchronous (frame-based) monitor path is supported. Time-based
   diary mode (``--diary-mode=time``) is forced back to frame mode in
   ``--no-ai``, since prompting from a background thread is awkward.
"""
from __future__ import annotations

import shutil
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np


DiaryAction = Literal[
    "continue", "stop", "override", "command", "ask_help", "force_converge",
]


@dataclass
class DiaryCheckResult:
    action: DiaryAction
    new_instruction: str
    reasoning: str
    diary_entry: str
    completion_pct: float = 0.0
    # Mirrors the live monitor: optional header used by interface.py's
    # _handle_ask_help when surfacing an ask_help to the operator. Falls
    # back to "MAX CORRECTIONS REACHED" downstream when empty.
    ask_help_header: str = ""


def _prompt(prompt: str, default: str = "") -> str:
    """input() that lets Ctrl-C actually break out.

    The hardware runner installs a SIGINT handler that just flips a flag
    without raising; under that handler, a Ctrl-C while ``input()`` is
    blocking is silently swallowed (the underlying read syscall restarts).
    Around the read we restore Python's default SIGINT handler, which raises
    KeyboardInterrupt, then put the runner's handler back. The exception is
    allowed to propagate so the caller's finally-block tears down hardware.
    """
    suffix = f" [{default}]" if default else ""
    prev = None
    try:
        prev = signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError):
        # Not on the main thread; fall through with whatever's installed.
        prev = None
    try:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            raw = ""
    finally:
        if prev is not None:
            try:
                signal.signal(signal.SIGINT, prev)
            except (ValueError, OSError):
                pass
    return raw or default


def _prompt_float(prompt: str, default: float) -> float:
    raw = _prompt(prompt, f"{default:.2f}")
    try:
        return float(raw)
    except ValueError:
        print(f"  [no-ai] '{raw}' is not a number, using {default}")
        return default


# ---------------------------------------------------------------------------
# Planner stubs
# ---------------------------------------------------------------------------

class ManualLLMUserInterface:
    """Stand-in for ``rvln.ai.llm_interface.LLMUserInterface``.

    Asks the operator to enumerate the subgoals that the LTL planner would
    normally produce, then exposes them via ``ltl_nl_formula`` in the same
    shape downstream code already reads.
    """

    def __init__(self, model: str = "no-ai", use_constraints: bool = False):
        self._model = model
        self._use_constraints = use_constraints
        self.ltl_nl_formula: Dict[str, Any] = {}
        self.llm_call_records: List[Dict[str, Any]] = []

    def make_natural_language_request(
        self, request: str, ignore_cache: bool = False,
    ) -> str:
        print()
        print("=" * 60)
        print("[no-ai] Manual LTL planning")
        print(f"  instruction: {request}")
        print("  Enter subgoals one per line, blank line to finish.")
        print("  These are the predicates pi_1, pi_2, ... in order.")
        print("=" * 60)
        predicates: Dict[str, str] = {}
        idx = 1
        while True:
            line = _prompt(f"  pi_{idx}")
            if not line:
                if not predicates:
                    print("  [no-ai] At least one subgoal is required.")
                    continue
                break
            predicates[f"pi_{idx}"] = line
            idx += 1

        formula = " & ".join(f"F {k}" for k in predicates.keys()) or "true"
        self.ltl_nl_formula = {
            "ltl_nl_formula": formula,
            "pi_predicates": predicates,
        }
        return formula

    def reset_to_baseline_context(self) -> None:
        return None


class ManualSequentialLTLPlanner:
    """Stand-in for ``rvln.ai.sequential_ltl_planner.SequentialLTLPlanner``.

    Walks through the manually entered predicates in order. Mirrors the public
    surface used by ``run_hardware``: ``plan_from_natural_language``,
    ``pi_map``, ``get_next_predicate``, ``advance_state``, ``finished``.
    """

    def __init__(self, llm_interface: ManualLLMUserInterface):
        self.llm_interface = llm_interface
        self.pi_map: Dict[str, str] = {}
        self._order: List[str] = []
        self._idx: int = 0
        self._last_returned_predicate_key: Optional[str] = None
        self.finished: bool = False

    def plan_from_natural_language(self, instruction: str) -> None:
        if not instruction or not isinstance(instruction, str):
            raise ValueError("Instruction must be a non-empty string.")
        self.llm_interface.make_natural_language_request(instruction)
        self.pi_map = dict(self.llm_interface.ltl_nl_formula.get("pi_predicates", {}))
        self._order = list(self.pi_map.keys())
        self._idx = 0
        self._last_returned_predicate_key = None
        self.finished = not self._order

    def get_next_predicate(self) -> Optional[str]:
        if self.finished or self._idx >= len(self._order):
            self.finished = True
            return None
        key = self._order[self._idx]
        self._last_returned_predicate_key = key
        return self.pi_map[key]

    def advance_state(self, finished_task_nl: str) -> None:
        self._idx += 1
        if self._idx >= len(self._order):
            self.finished = True


# ---------------------------------------------------------------------------
# Subgoal converter stub
# ---------------------------------------------------------------------------

@dataclass
class ConversionResult:
    instruction: str


class ManualSubgoalConverter:
    """Pass-through subgoal converter.

    Prompts the operator for the OpenVLA-compatible imperative form of the
    subgoal, defaulting to the subgoal text itself. Empty input keeps the
    default. Mirrors the field surface used downstream
    (``llm_call_records``, ``convert``, ``ConversionResult.instruction``).
    """

    def __init__(self, model: str = "no-ai"):
        self._model = model
        self.llm_call_records: List[Dict[str, Any]] = []

    def convert(self, subgoal: str) -> ConversionResult:
        print()
        print(f"[no-ai] Subgoal converter for: '{subgoal}'")
        print("  Enter OpenVLA instruction (blank = pass-through).")
        instruction = _prompt("  openvla", default=subgoal)
        return ConversionResult(instruction=instruction)


# ---------------------------------------------------------------------------
# Goal-adherence monitor stub
# ---------------------------------------------------------------------------

class ManualGoalAdherenceMonitor:
    """Terminal-driven stand-in for ``GoalAdherenceMonitor`` (sync mode only).

    At each ``check_interval`` frame and on every convergence the operator
    is prompted for the action, completion percentage, and (if needed) a
    correction instruction. Async/time-based mode is not supported here;
    callers must run with ``--diary-mode=frame``.
    """

    def __init__(
        self,
        subgoal: str,
        check_interval: int,
        model: str = "no-ai",
        artifacts_dir: Optional[Path] = None,
        max_corrections: int = 3,
        check_interval_s: Optional[float] = None,
        stall_window: int = 10,
        stall_threshold: float = 0.05,
        stall_completion_floor: float = 0.8,
        constraints: Optional[List[Any]] = None,
        negative_constraints: Optional[List[str]] = None,
        global_backend: str = "vlm_grid",
        global_model: Optional[str] = None,
        single_frame_mode: bool = False,
    ):
        if check_interval_s is not None:
            raise ValueError(
                "ManualGoalAdherenceMonitor does not support time-based mode. "
                "Run with --diary-mode=frame when using --no-ai."
            )
        self._subgoal = subgoal
        self._check_interval = max(1, int(check_interval))
        self._model = model
        self._artifacts_dir = artifacts_dir
        self._max_corrections = max_corrections

        self._diary: List[str] = []
        self._step = 0
        self._corrections_used = 0
        self._parse_failures = 0
        self._vlm_calls = 0
        self._vlm_rtts: List[Dict[str, Any]] = []
        self._last_completion_pct: float = 0.0
        self._peak_completion: float = 0.0
        self._last_displacement: List[float] = [0.0, 0.0, 0.0, 0.0]
        self._temp_dir: Optional[str] = None

    # ----- public properties matching GoalAdherenceMonitor -----------------

    @property
    def diary(self) -> List[str]:
        return list(self._diary)

    @property
    def step(self) -> int:
        return self._step

    @property
    def corrections_used(self) -> int:
        return self._corrections_used

    @property
    def max_corrections(self) -> int:
        return self._max_corrections

    def reset_grace_state(self) -> None:
        """Mirror of GoalAdherenceMonitor.reset_grace_state. The stub has
        no async checkpoint history but does track _corrections_used, so
        we reset that. interface.py calls this when the operator picks
        'continue' so the next prompt does not re-fire instantly."""
        self._corrections_used = 0

    @property
    def corrections_exhausted(self) -> bool:
        return self._corrections_used >= self._max_corrections

    @property
    def last_completion_pct(self) -> float:
        return self._last_completion_pct

    @property
    def peak_completion(self) -> float:
        return self._peak_completion

    @property
    def parse_failures(self) -> int:
        return self._parse_failures

    @property
    def vlm_calls(self) -> int:
        return self._vlm_calls

    @property
    def vlm_rtts(self) -> List[Dict[str, Any]]:
        return list(self._vlm_rtts)

    # ----- public methods --------------------------------------------------

    def poll_result(self) -> Optional[DiaryCheckResult]:
        return None

    def request_convergence(
        self, frame_path: Union[Path, str], displacement: List[float],
    ) -> None:
        # Not used in sync mode but kept for surface compatibility.
        return None

    def on_frame(
        self,
        frame_image_or_path: Union[np.ndarray, Path, str],
        displacement: Optional[List[float]] = None,
    ) -> DiaryCheckResult:
        self._step += 1
        if displacement is not None:
            self._last_displacement = list(displacement)

        if self._step % self._check_interval != 0:
            return DiaryCheckResult(
                action="continue",
                new_instruction="",
                reasoning="",
                diary_entry="",
                completion_pct=self._last_completion_pct,
            )
        return self._prompt_checkpoint(frame_image_or_path)

    def on_convergence(
        self,
        latest_frame: Union[np.ndarray, Path, str],
        displacement: Optional[List[float]] = None,
    ) -> DiaryCheckResult:
        if displacement is not None:
            self._last_displacement = list(displacement)

        if self._corrections_used >= self._max_corrections:
            return DiaryCheckResult(
                action="ask_help",
                new_instruction="",
                reasoning=f"Max corrections ({self._max_corrections}) exhausted.",
                diary_entry="",
                completion_pct=self._last_completion_pct,
            )

        print()
        print("-" * 60)
        print(f"[no-ai] Convergence check: subgoal '{self._subgoal}'")
        self._print_state(latest_frame)
        print("  Choose action:")
        print("    [s] stop          subgoal complete")
        print("    [c] command       send a corrective instruction")
        print("    [h] ask_help      hand off to operator (max corrections)")
        choice = _prompt("  action", default="s").lower()

        if choice.startswith("c"):
            new_instr = _prompt("  corrective instruction")
            if not new_instr:
                print("  [no-ai] Empty corrective instruction; treating as stop.")
                self._update_completion(1.0)
                return DiaryCheckResult(
                    action="stop", new_instruction="",
                    reasoning="manual: empty corrective -> stop",
                    diary_entry="", completion_pct=self._last_completion_pct,
                )
            pct = _prompt_float("  completion_pct (0-1)", self._last_completion_pct)
            self._update_completion(pct)
            self._corrections_used += 1
            return DiaryCheckResult(
                action="command", new_instruction=new_instr,
                reasoning="manual convergence command",
                diary_entry="", completion_pct=self._last_completion_pct,
            )
        if choice.startswith("h"):
            return DiaryCheckResult(
                action="ask_help", new_instruction="",
                reasoning="manual: ask_help",
                diary_entry="", completion_pct=self._last_completion_pct,
            )
        # default: stop
        self._update_completion(1.0)
        return DiaryCheckResult(
            action="stop", new_instruction="",
            reasoning="manual convergence stop",
            diary_entry="", completion_pct=self._last_completion_pct,
        )

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_dir = None

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    # ----- internals -------------------------------------------------------

    def _prompt_checkpoint(
        self, frame_image_or_path: Union[np.ndarray, Path, str],
    ) -> DiaryCheckResult:
        print()
        print("-" * 60)
        print(f"[no-ai] Checkpoint at step {self._step}: subgoal '{self._subgoal}'")
        self._print_state(frame_image_or_path)
        print("  Choose action:")
        print("    [c] continue       (default)")
        print("    [s] stop           subgoal complete")
        print("    [f] force_converge run convergence check now")
        print("    [h] ask_help       hand off to operator")
        print("    [o] override       send a new OpenVLA instruction")
        choice = _prompt("  action", default="c").lower()

        diary_entry = _prompt("  diary entry (optional)") or ""
        if diary_entry:
            self._diary.append(diary_entry)

        if choice.startswith("s"):
            self._update_completion(1.0)
            return DiaryCheckResult(
                action="stop", new_instruction="",
                reasoning="manual: stop",
                diary_entry=diary_entry, completion_pct=self._last_completion_pct,
            )
        if choice.startswith("f"):
            pct = _prompt_float("  completion_pct (0-1)", self._last_completion_pct)
            self._update_completion(pct)
            return DiaryCheckResult(
                action="force_converge", new_instruction="",
                reasoning="manual: force_converge",
                diary_entry=diary_entry, completion_pct=self._last_completion_pct,
            )
        if choice.startswith("h"):
            pct = _prompt_float("  completion_pct (0-1)", self._last_completion_pct)
            self._update_completion(pct)
            return DiaryCheckResult(
                action="ask_help", new_instruction="",
                reasoning="manual: ask_help",
                diary_entry=diary_entry, completion_pct=self._last_completion_pct,
            )
        if choice.startswith("o"):
            new_instr = _prompt("  new instruction")
            if not new_instr:
                print("  [no-ai] Empty instruction; falling back to continue.")
                pct = _prompt_float("  completion_pct (0-1)", self._last_completion_pct)
                self._update_completion(pct)
                return DiaryCheckResult(
                    action="continue", new_instruction="",
                    reasoning="manual: empty override -> continue",
                    diary_entry=diary_entry, completion_pct=self._last_completion_pct,
                )
            pct = _prompt_float("  completion_pct (0-1)", self._last_completion_pct)
            self._update_completion(pct)
            return DiaryCheckResult(
                action="override", new_instruction=new_instr,
                reasoning="manual: override",
                diary_entry=diary_entry, completion_pct=self._last_completion_pct,
            )
        pct = _prompt_float("  completion_pct (0-1)", self._last_completion_pct)
        self._update_completion(pct)
        return DiaryCheckResult(
            action="continue", new_instruction="",
            reasoning="manual: continue",
            diary_entry=diary_entry, completion_pct=self._last_completion_pct,
        )

    def _print_state(self, frame_ref: Union[np.ndarray, Path, str]) -> None:
        d = self._last_displacement
        print(
            f"  displacement: x={d[0] / 100:.2f} m, y={d[1] / 100:.2f} m, "
            f"z={d[2] / 100:.2f} m, yaw={d[3]:.1f} deg"
        )
        print(f"  prev completion: {self._last_completion_pct:.2f}")
        if self._diary:
            tail = self._diary[-3:]
            print(f"  recent diary ({len(tail)}/{len(self._diary)}):")
            for line in tail:
                print(f"    - {line}")
        if isinstance(frame_ref, (Path, str)):
            print(f"  latest frame: {frame_ref}")

    def _update_completion(self, pct: float) -> None:
        pct = max(0.0, min(1.0, float(pct)))
        self._last_completion_pct = pct
        self._peak_completion = max(self._peak_completion, pct)
