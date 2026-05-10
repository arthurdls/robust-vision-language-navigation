"""Tests for scripts/summarize_results.py annotation-collision guard."""

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summarize_results.py"
_spec = importlib.util.spec_from_file_location("summarize_results", _SCRIPT)
sr = importlib.util.module_from_spec(_spec)
sys.modules["summarize_results"] = sr
_spec.loader.exec_module(sr)


def _row(cond, **fields):
    base = {
        "condition": cond,
        "label": f"C{cond}",
        "task_success": None,
        "subgoals_completed": None,
        "subgoals_total": None,
        "notes": "",
    }
    base.update(fields)
    return base


class TestAnnotationCollisions:
    def test_no_existing_data_no_collision(self):
        existing = {"rows": [_row(0)]}
        new = {"rows": [_row(0)]}
        assert sr.annotation_collisions(existing, new) == []

    def test_filling_in_empty_fields_is_not_collision(self):
        existing = {"rows": [_row(0)]}
        new = {"rows": [_row(0, task_success=1, notes="ok")]}
        assert sr.annotation_collisions(existing, new) == []

    def test_erasing_task_success_is_collision(self):
        existing = {"rows": [_row(0, task_success=1)]}
        new = {"rows": [_row(0, task_success=None)]}
        issues = sr.annotation_collisions(existing, new)
        assert len(issues) == 1
        assert "C0 task_success" in issues[0]

    def test_erasing_notes_is_collision(self):
        existing = {"rows": [_row(0, notes="full system completed all subgoals")]}
        new = {"rows": [_row(0, notes="")]}
        issues = sr.annotation_collisions(existing, new)
        assert len(issues) == 1
        assert "C0 notes" in issues[0]

    def test_changing_value_is_collision(self):
        existing = {"rows": [_row(0, task_success=1)]}
        new = {"rows": [_row(0, task_success=0)]}
        issues = sr.annotation_collisions(existing, new)
        assert len(issues) == 1

    def test_unchanged_value_is_not_collision(self):
        existing = {"rows": [_row(0, task_success=1, notes="x")]}
        new = {"rows": [_row(0, task_success=1, notes="x")]}
        assert sr.annotation_collisions(existing, new) == []

    def test_multiple_conditions_reported_separately(self):
        existing = {"rows": [_row(0, task_success=1), _row(1, notes="hi")]}
        new = {"rows": [_row(0, task_success=None), _row(1, notes="")]}
        issues = sr.annotation_collisions(existing, new)
        assert len(issues) == 2
        assert any("C0" in i for i in issues)
        assert any("C1" in i for i in issues)

    def test_new_condition_missing_from_existing_ignored(self):
        existing = {"rows": [_row(0, task_success=1)]}
        new = {"rows": [_row(0, task_success=1), _row(1)]}
        assert sr.annotation_collisions(existing, new) == []
