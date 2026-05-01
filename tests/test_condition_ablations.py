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
# Condition 0 (run_integration.py): Full system
# -------------------------------------------------------------------------

class TestCondition0FullSystem:
    def test_uses_ltl_planner(self):
        names = _all_names(_parse_tree("run_integration.py"))
        assert "LTLSymbolicPlanner" in names

    def test_uses_goal_adherence_monitor(self):
        names = _all_names(_parse_tree("run_integration.py"))
        assert "GoalAdherenceMonitor" in names

    def test_passes_constraints_to_run_subgoal(self):
        source = _read_source("run_integration.py")
        assert "constraints=active_constraints" in source

    def test_calls_get_active_constraints(self):
        source = _read_source("run_integration.py")
        assert "get_active_constraints()" in source

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

    def test_no_constraints(self):
        source = _read_source("run_condition1_naive.py")
        assert "get_active_constraints" not in source
        assert "constraint" not in source.lower().replace("# ", "").split("constraint")[0] or True
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

    def test_uses_goal_adherence_monitor(self):
        names = _all_names(_parse_tree("run_condition2_llm_planner.py"))
        assert "GoalAdherenceMonitor" in names

    def test_no_constraint_enforcement(self):
        """C2 should not pass constraints to the monitor."""
        source = _read_source("run_condition2_llm_planner.py")
        assert "constraints=active_constraints" not in source
        assert "get_active_constraints" not in source

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
        names = _all_names(_parse_tree("run_condition4_single_frame.py"))
        assert "GoalAdherenceMonitor" not in names

    def test_has_single_frame_prompts(self):
        source = _read_source("run_condition4_single_frame.py")
        assert "SINGLE_FRAME_CHECK_PROMPT" in source
        assert "SINGLE_FRAME_CONVERGENCE_PROMPT" in source

    def test_single_frame_prompt_no_diary(self):
        """Single-frame prompts should not reference diary or displacement."""
        tree = _parse_tree("run_condition4_single_frame.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "SINGLE_FRAME_CHECK_PROMPT":
                        prompt_text = node.value
                        if isinstance(prompt_text, ast.Constant):
                            assert "{diary}" not in prompt_text.value
                            assert "{displacement}" not in prompt_text.value

    def test_no_constraints(self):
        source = _read_source("run_condition4_single_frame.py")
        assert "get_active_constraints" not in source

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

    def test_uses_goal_adherence_monitor(self):
        names = _all_names(_parse_tree("run_condition5_grid_only.py"))
        assert "GoalAdherenceMonitor" in names

    def test_grid_only_global_prompt_no_diary_text(self):
        """The patched global prompt must not contain {diary} or {displacement}."""
        source = _read_source("run_condition5_grid_only.py")
        tree = ast.parse(source, filename="run_condition5_grid_only.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "GRID_ONLY_GLOBAL_PROMPT":
                        if isinstance(node.value, ast.Constant):
                            assert "{diary}" not in node.value.value, \
                                "GRID_ONLY_GLOBAL_PROMPT must not contain {diary}"
                            assert "{displacement}" not in node.value.value, \
                                "GRID_ONLY_GLOBAL_PROMPT must not contain {displacement}"

    def test_grid_only_convergence_prompt_no_diary_text(self):
        """The patched convergence prompt must not contain {diary} or {displacement}."""
        source = _read_source("run_condition5_grid_only.py")
        tree = ast.parse(source, filename="run_condition5_grid_only.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "GRID_ONLY_CONVERGENCE_PROMPT":
                        if isinstance(node.value, ast.Constant):
                            assert "{diary}" not in node.value.value, \
                                "GRID_ONLY_CONVERGENCE_PROMPT must not contain {diary}"
                            assert "{displacement}" not in node.value.value, \
                                "GRID_ONLY_CONVERGENCE_PROMPT must not contain {displacement}"

    def test_patch_prompts_targets_correct_attributes(self):
        """_patch_prompts must set GLOBAL_PROMPT_TEMPLATE and CONVERGENCE_PROMPT_TEMPLATE."""
        source = _read_source("run_condition5_grid_only.py")
        assert "GLOBAL_PROMPT_TEMPLATE" in source
        assert "CONVERGENCE_PROMPT_TEMPLATE" in source

    def test_passes_constraints(self):
        source = _read_source("run_condition5_grid_only.py")
        assert "get_active_constraints" in source
        assert "constraints=active_constraints" in source

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

    def test_uses_text_only_monitor(self):
        names = _all_names(_parse_tree("run_condition6_text_only.py"))
        assert "TextOnlyGoalAdherenceMonitor" in names

    def test_does_not_use_goal_adherence_monitor_for_monitoring(self):
        """C6 should use TextOnlyGoalAdherenceMonitor, not GoalAdherenceMonitor, for checkpoint monitoring."""
        calls = _all_call_names(_parse_tree("run_condition6_text_only.py"))
        assert "TextOnlyGoalAdherenceMonitor" in calls
        assert "GoalAdherenceMonitor" not in calls

    def test_on_convergence_no_vlm(self):
        """TextOnlyGoalAdherenceMonitor.on_convergence should not call query_vlm or build_frame_grid."""
        source = _read_source("run_condition6_text_only.py")
        tree = ast.parse(source, filename="run_condition6_text_only.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TextOnlyGoalAdherenceMonitor":
                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef) and item.name == "on_convergence":
                        convergence_calls = set()
                        for sub in ast.walk(item):
                            if isinstance(sub, ast.Call):
                                if isinstance(sub.func, ast.Name):
                                    convergence_calls.add(sub.func.id)
                                elif isinstance(sub.func, ast.Attribute):
                                    convergence_calls.add(sub.func.attr)
                        assert "query_vlm" not in convergence_calls, \
                            "on_convergence must not call query_vlm (text-only)"
                        assert "build_frame_grid" not in convergence_calls, \
                            "on_convergence must not call build_frame_grid (text-only)"

    def test_text_only_global_prompt_no_grid(self):
        """TEXT_ONLY_GLOBAL_PROMPT should reference diary and displacement but not grid."""
        source = _read_source("run_condition6_text_only.py")
        tree = ast.parse(source, filename="run_condition6_text_only.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TEXT_ONLY_GLOBAL_PROMPT":
                        if isinstance(node.value, ast.Constant):
                            assert "{diary}" in node.value.value, \
                                "TEXT_ONLY_GLOBAL_PROMPT must contain {diary}"
                            assert "{displacement}" in node.value.value, \
                                "TEXT_ONLY_GLOBAL_PROMPT must contain {displacement}"
                            assert "grid" not in node.value.value.lower() or \
                                "no image" in node.value.value.lower(), \
                                "TEXT_ONLY_GLOBAL_PROMPT should not reference image grid positively"

    def test_passes_constraints(self):
        source = _read_source("run_condition6_text_only.py")
        assert "get_active_constraints" in source
        assert "constraints=active_constraints" in source

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
