#!/usr/bin/env python3
"""
Summarize the 7-episode preliminary evaluation.

Reads, for every condition, the most-recent run_info.json under
results/condition{0..6}/<map>/<run_dir>/run_info.json, and the
human-review CSV results/annotations.csv (one row per episode), and
emits a single per-condition outcome table plus M6-M8 telemetry.

There is one episode per condition (n=1) under the rescoped preliminary
design (experimental_design.txt Section 8a), so this script reports
raw values only: no Wilson confidence intervals, no Fisher's exact, no
McNemar, no "significant" flags. Reviewers should look at the per-
condition row plus the qualitative video review.

annotations.csv schema (filled in by hand after each episode):

    condition,task_success,subgoals_completed,subgoals_total,notes
    0,1,6,6,full system completed all subgoals
    1,0,1,6,naive OpenVLA stopped after the first instruction step
    ...

Usage:
    python scripts/summarize_results.py [--results_dir results]

Writes:
    <results_dir>/analysis_output.json  (machine-readable)
    stdout                               (human-readable table)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


CONDITION_LABELS: Dict[int, str] = {
    0: "C0 full system",
    1: "C1 naive (no decomp)",
    2: "C2 LLM planner (no LTL)",
    3: "C3 open-loop LTL",
    4: "C4 single-frame monitor",
    5: "C5 grid-only monitor",
    6: "C6 text-only global",
}


def find_latest_run_info(condition_dir: Path) -> Optional[Path]:
    """Return the most recently modified run_info.json under condition_dir, or None.

    Walks one level deep (results/condition{N}/<map>/<run_dir>/run_info.json).
    """
    if not condition_dir.is_dir():
        return None
    candidates: List[Path] = list(condition_dir.glob("*/*/run_info.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_run_info(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def load_annotations(csv_path: Path) -> Dict[int, Dict[str, Any]]:
    """Return {condition_int: row_dict}. Missing or empty fields stay as None."""
    out: Dict[int, Dict[str, Any]] = {}
    if not csv_path.is_file():
        return out
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cond = int(row["condition"])
            except (KeyError, TypeError, ValueError):
                continue
            out[cond] = {
                "task_success": _maybe_int(row.get("task_success")),
                "subgoals_completed": _maybe_int(row.get("subgoals_completed")),
                "subgoals_total": _maybe_int(row.get("subgoals_total")),
                "notes": (row.get("notes") or "").strip(),
            }
    return out


def _maybe_int(s: Any) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def telemetry_from_run_info(run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Pull M6-M8 raw values out of a run_info.json. Returns 0/None for missing fields."""
    monitor_rtts = [
        r.get("rtt_s")
        for r in run_info.get("vlm_call_records", [])
        if isinstance(r, dict)
        and isinstance(r.get("label", ""), str)
        and r.get("label", "").startswith(("local", "global", "convergence", "text_only"))
        and r.get("rtt_s") is not None
    ]
    avg_monitor_rtt = (
        round(sum(monitor_rtts) / len(monitor_rtts), 3) if monitor_rtts else None
    )
    return {
        "M6_monitor_calls": len(monitor_rtts),
        "M6_avg_monitor_rtt_s": avg_monitor_rtt,
        "M7_total_input_tokens": run_info.get("total_input_tokens", 0),
        "M7_total_output_tokens": run_info.get("total_output_tokens", 0),
        "M7_total_image_tokens": run_info.get("total_image_tokens", 0),
        "M8_total_steps": run_info.get("total_steps", 0),
        "M8_wall_clock_seconds": run_info.get("wall_clock_seconds", 0),
        "stop_reason": run_info.get("stop_reason", ""),
        "aborted": run_info.get("aborted", False),
        "completed": run_info.get("completed", False),
    }


def summarize(results_dir: Path) -> Dict[str, Any]:
    annotations = load_annotations(results_dir / "annotations.csv")
    rows: List[Dict[str, Any]] = []
    for cond in sorted(CONDITION_LABELS.keys()):
        cond_dir = results_dir / f"condition{cond}"
        run_info_path = find_latest_run_info(cond_dir)
        run_info = load_run_info(run_info_path) if run_info_path else None
        ann = annotations.get(cond, {})
        telemetry = telemetry_from_run_info(run_info) if run_info else {}
        rows.append({
            "condition": cond,
            "label": CONDITION_LABELS[cond],
            "run_info_path": str(run_info_path) if run_info_path else None,
            "task_success": ann.get("task_success"),
            "subgoals_completed": ann.get("subgoals_completed"),
            "subgoals_total": ann.get("subgoals_total"),
            "notes": ann.get("notes", ""),
            **telemetry,
        })
    return {
        "design": "preliminary_n1_per_condition",
        "results_dir": str(results_dir),
        "annotations_csv": str(results_dir / "annotations.csv"),
        "rows": rows,
    }


def _fmt(val: Any, default: str = "-") -> str:
    if val is None:
        return default
    return str(val)


def print_table(summary: Dict[str, Any]) -> None:
    rows = summary["rows"]
    header = (
        "Cond  Label                       Outcome  Subgoals       Steps    Wall(s)   "
        "Monitor calls  Avg RTT(s)  Tokens(in/out/img)         Stop reason"
    )
    sep = "-" * len(header)
    print()
    print(f"PRELIMINARY RESULTS (n=1 per condition)")
    print(f"results_dir = {summary['results_dir']}")
    print()
    print(header)
    print(sep)
    for row in rows:
        ts = row.get("task_success")
        outcome = "PASS" if ts == 1 else ("FAIL" if ts == 0 else "?")
        sg = (
            f"{_fmt(row.get('subgoals_completed'), '?')}/"
            f"{_fmt(row.get('subgoals_total'), '?')}"
        )
        tokens = (
            f"{_fmt(row.get('M7_total_input_tokens'), '0')}/"
            f"{_fmt(row.get('M7_total_output_tokens'), '0')}/"
            f"{_fmt(row.get('M7_total_image_tokens'), '0')}"
        )
        print(
            f"C{row['condition']}    {row['label']:<26}  "
            f"{outcome:<7}  {sg:<13}  "
            f"{_fmt(row.get('M8_total_steps'), '-'):>5}    "
            f"{_fmt(row.get('M8_wall_clock_seconds'), '-'):>7}   "
            f"{_fmt(row.get('M6_monitor_calls'), '-'):>13}  "
            f"{_fmt(row.get('M6_avg_monitor_rtt_s'), '-'):>10}  "
            f"{tokens:<25}  "
            f"{_fmt(row.get('stop_reason'), '-')}"
        )
    print(sep)
    print()
    print("Notes per Section 8a/11 of experimental_design.txt:")
    print("  - This is a preliminary, single-task evaluation (n=1 per condition).")
    print("  - No confidence intervals or hypothesis tests are reported (n=1 makes them")
    print("    uninformative). Read the per-row outcome plus the qualitative video review.")
    print("  - C3 deliberately reports subgoals_total = total LTL subgoals processed,")
    print("    not the index of failure (Section 8c, C3 abort exception).")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize 7-episode preliminary results.")
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Path to the top-level results directory (default: results).",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Where to write the JSON summary (default: <results_dir>/analysis_output.json).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"results_dir not found: {results_dir}", file=sys.stderr)
        return 2

    summary = summarize(results_dir)
    print_table(summary)

    out_path = Path(args.output_json) if args.output_json else results_dir / "analysis_output.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote machine-readable summary to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
