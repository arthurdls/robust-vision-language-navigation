"""
Structural ablation tests for condition scripts.

Verifies each condition script includes/excludes the right components per the
experimental design (Section 8c ablation table). These tests catch accidental
inclusion of monitoring in open-loop conditions, missing constraint pass-through,
prompt template leaks, etc.

No external dependencies (no simulator, no API keys, no GPU).

Run: conda run -n rvln-sim pytest tests/test_condition_ablations.py -v
"""

import ast
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"


def _read_source(filename: str) -> str:
    return (_SCRIPTS / filename).read_text()


def _parse_tree(filename: str) -> ast.Module:
    return ast.parse(_read_source(filename), filename=filename)


def _all_names(tree: ast.Module) -> set:
    """Collect every Name and Attribute id referenced in the AST."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _all_string_literals(tree: ast.Module) -> list:
    """Collect every string constant in the AST."""
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
    return strings


def _all_call_names(tree: ast.Module) -> set:
    """Collect names of all function/class calls (Name nodes in Call.func)."""
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return calls


# -------------------------------------------------------------------------
# Shared: no condition script should reference the old constraint plumbing
# -------------------------------------------------------------------------

ALL_CONDITION_SCRIPTS = [
    "run_integration.py",
    "run_condition1_naive.py",
    "run_condition2_llm_planner.py",
    "run_condition3_open_loop.py",
    "run_condition4_single_frame.py",
    "run_condition5_grid_only.py",
    "run_condition6_text_only.py",
]


class TestNoConstraintPlumbing:
    """After the LTL-NL prompt unification, no script should call
    get_active_constraints, pass constraints=active_constraints, or
    reference use_constraints."""

    def test_no_get_active_constraints(self):
        for fname in ALL_CONDITION_SCRIPTS:
            source = _read_source(fname)
            assert "get_active_constraints" not in source, \
                f"{fname} still references get_active_constraints"

    def test_no_constraints_kwarg(self):
        for fname in ALL_CONDITION_SCRIPTS:
            source = _read_source(fname)
            assert "constraints=active_constraints" not in source, \
                f"{fname} still passes constraints=active_constraints"

    def test_no_use_constraints(self):
        for fname in ALL_CONDITION_SCRIPTS:
            source = _read_source(fname)
            assert "use_constraints" not in source, \
                f"{fname} still references use_constraints"

    def test_no_sequential_ltl_planner(self):
        for fname in ALL_CONDITION_SCRIPTS:
            names = _all_names(_parse_tree(fname))
            assert "SequentialLTLPlanner" not in names, \
                f"{fname} still references SequentialLTLPlanner"


# -------------------------------------------------------------------------
# Condition 0 (run_integration.py): Full system
# -------------------------------------------------------------------------

class TestCondition0FullSystem:
    def test_uses_ltl_planner(self):
        names = _all_names(_parse_tree("run_integration.py"))
        assert "LTLSymbolicPlanner" in names

    def test_uses_full_monitor_mode(self):
        source = _read_source("run_integration.py")
        assert 'monitor_mode="full"' in source

    def test_condition_label(self):
        source = _read_source("run_integration.py")
        assert '"condition0_full_system"' in source


# -------------------------------------------------------------------------
# Condition 1 (run_condition1_naive.py): No decomposition, no monitoring
# -------------------------------------------------------------------------

class TestCondition1Naive:
    def test_no_ltl_planner(self):
        names = _all_names(_parse_tree("run_condition1_naive.py"))
        assert "LTLSymbolicPlanner" not in names

    def test_no_goal_adherence_monitor(self):
        names = _all_names(_parse_tree("run_condition1_naive.py"))
        assert "GoalAdherenceMonitor" not in names

    def test_no_subgoal_converter(self):
        names = _all_names(_parse_tree("run_condition1_naive.py"))
        assert "SubgoalConverter" not in names

    def test_no_spot_import(self):
        source = _read_source("run_condition1_naive.py")
        assert "import spot" not in source

    def test_no_constraint_info(self):
        names = _all_names(_parse_tree("run_condition1_naive.py"))
        assert "ConstraintInfo" not in names

    def test_condition_label(self):
        source = _read_source("run_condition1_naive.py")
        assert '"condition1_naive"' in source


# -------------------------------------------------------------------------
# Condition 2 (run_condition2_llm_planner.py): LLM planner, no LTL
# -------------------------------------------------------------------------

class TestCondition2LLMPlanner:
    def test_no_ltl_planner(self):
        names = _all_names(_parse_tree("run_condition2_llm_planner.py"))
        assert "LTLSymbolicPlanner" not in names

    def test_no_spot(self):
        source = _read_source("run_condition2_llm_planner.py")
        assert "import spot" not in source

    def test_uses_full_monitor_mode(self):
        source = _read_source("run_condition2_llm_planner.py")
        assert 'monitor_mode="full"' in source

    def test_has_llm_decompose(self):
        source = _read_source("run_condition2_llm_planner.py")
        assert "_llm_decompose" in source

    def test_condition_label(self):
        source = _read_source("run_condition2_llm_planner.py")
        assert '"condition2_llm_planner"' in source


# -------------------------------------------------------------------------
# Condition 3 (run_condition3_open_loop.py): LTL but no monitor
# -------------------------------------------------------------------------

class TestCondition3OpenLoop:
    def test_uses_ltl_planner(self):
        names = _all_names(_parse_tree("run_condition3_open_loop.py"))
        assert "LTLSymbolicPlanner" in names

    def test_no_goal_adherence_monitor(self):
        names = _all_names(_parse_tree("run_condition3_open_loop.py"))
        assert "GoalAdherenceMonitor" not in names
        assert "TextOnlyGoalAdherenceMonitor" not in names

    def test_no_on_convergence(self):
        source = _read_source("run_condition3_open_loop.py")
        assert "on_convergence" not in source

    def test_no_on_frame(self):
        source = _read_source("run_condition3_open_loop.py")
        assert "on_frame" not in source

    def test_uses_subgoal_converter(self):
        names = _all_names(_parse_tree("run_condition3_open_loop.py"))
        assert "SubgoalConverter" in names

    def test_condition_label(self):
        source = _read_source("run_condition3_open_loop.py")
        assert '"condition3_open_loop"' in source


# -------------------------------------------------------------------------
# Condition 4 (run_condition4_single_frame.py): Single-frame VLM, no diary
# -------------------------------------------------------------------------

class TestCondition4SingleFrame:
    def test_uses_ltl_planner(self):
        names = _all_names(_parse_tree("run_condition4_single_frame.py"))
        assert "LTLSymbolicPlanner" in names

    def test_no_goal_adherence_monitor(self):
        """C4 uses its own single-frame loop, not GoalAdherenceMonitor."""
        calls = _all_call_names(_parse_tree("run_condition4_single_frame.py"))
        assert "GoalAdherenceMonitor" not in calls

    def test_uses_single_frame_monitor_mode(self):
        source = _read_source("run_condition4_single_frame.py")
        assert 'monitor_mode="single_frame"' in source

    def test_condition_label(self):
        source = _read_source("run_condition4_single_frame.py")
        assert '"condition4_single_frame"' in source


# -------------------------------------------------------------------------
# Condition 5 (run_condition5_grid_only.py): Grid only, no text diary
# -------------------------------------------------------------------------

class TestCondition5GridOnly:
    def test_uses_ltl_planner(self):
        names = _all_names(_parse_tree("run_condition5_grid_only.py"))
        assert "LTLSymbolicPlanner" in names

    def test_uses_grid_only_monitor_mode(self):
        source = _read_source("run_condition5_grid_only.py")
        assert 'monitor_mode="grid_only"' in source

    def test_condition_label(self):
        source = _read_source("run_condition5_grid_only.py")
        assert '"condition5_grid_only"' in source


# -------------------------------------------------------------------------
# Condition 6 (run_condition6_text_only.py): Text diary only, no image grid
# -------------------------------------------------------------------------

class TestCondition6TextOnly:
    def test_uses_ltl_planner(self):
        names = _all_names(_parse_tree("run_condition6_text_only.py"))
        assert "LTLSymbolicPlanner" in names

    def test_uses_text_only_monitor_mode(self):
        source = _read_source("run_condition6_text_only.py")
        assert 'monitor_mode="text_only"' in source

    def test_condition_label(self):
        source = _read_source("run_condition6_text_only.py")
        assert '"condition6_text_only"' in source


# -------------------------------------------------------------------------
# Cross-condition output schema checks
# -------------------------------------------------------------------------

class TestOutputSchemaConsistency:
    CONDITIONS_WITH_SUBGOALS = [
        "run_condition2_llm_planner.py",
        "run_condition3_open_loop.py",
        "run_condition4_single_frame.py",
        "run_condition5_grid_only.py",
        "run_condition6_text_only.py",
        "run_integration.py",
    ]

    def test_all_conditions_log_subgoal_summaries(self):
        for fname in self.CONDITIONS_WITH_SUBGOALS:
            source = _read_source(fname)
            assert "subgoal_summaries" in source, \
                f"{fname} missing 'subgoal_summaries' in run_info"

    def test_condition1_has_no_subgoal_summaries(self):
        source = _read_source("run_condition1_naive.py")
        assert "subgoal_summaries" not in source

    def test_all_conditions_log_total_steps(self):
        all_scripts = self.CONDITIONS_WITH_SUBGOALS + ["run_condition1_naive.py"]
        for fname in all_scripts:
            source = _read_source(fname)
            assert '"total_steps"' in source, \
                f"{fname} missing 'total_steps' in run_info"

    def test_all_conditions_save_trajectory_log(self):
        all_scripts = self.CONDITIONS_WITH_SUBGOALS + ["run_condition1_naive.py"]
        for fname in all_scripts:
            source = _read_source(fname)
            assert "trajectory_log.json" in source, \
                f"{fname} missing trajectory_log.json save"
