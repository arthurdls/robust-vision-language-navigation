#!/usr/bin/env python3
"""
Analyze human-annotated experimental results from annotations.csv.

Reads results/annotations.csv (human video review judgments) and computes:
  - M1: Task Success Rate per condition (Wilson CI)
  - M3: Subgoal Success Rate per condition (Wilson CI)
  - Per-task breakdown table (3 tasks x 7 conditions)
  - Task-level pass rate (majority of variants succeeded)
  - Fisher's exact test for primary comparisons (C0 vs C3, C0 vs C1)
  - McNemar's exact binomial for paired comparisons
  - Secondary comparisons reported descriptively (no hypothesis tests)

Optionally reads run_info.json files for diagnostic metrics (M4, M6, M7, M8)
when --results_dir is provided.

Statistical design:
  - 2 pre-specified primary comparisons at unadjusted alpha = 0.05
  - No Bonferroni correction (n=9 is too small for any correction to be useful)
  - Per-task breakdown table is the primary evidence
  - Episode-level p-values supplement, not replace, the descriptive results
  - Effective sample size is ~3 (3 tasks with 3 near-identical variants each)
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


CONDITION_ORDER = [
    "condition0", "condition1", "condition2", "condition3",
    "condition4", "condition5", "condition6",
]

CONDITION_LABELS = {
    "condition0": "C0: Full System",
    "condition1": "C1: Naive End-to-End",
    "condition2": "C2: LLM Planner (No LTL)",
    "condition3": "C3: Open Loop (No Monitor)",
    "condition4": "C4: Single-Frame Monitor",
    "condition5": "C5: Grid-Only Monitor",
    "condition6": "C6: Text-Only Global",
}

PRIMARY_COMPARISONS = [
    ("condition0", "condition3"),  # Monitoring ablation (headline)
    ("condition0", "condition1"),  # Decomposition necessity
]

SECONDARY_COMPARISONS = [
    ("condition0", "condition2"),
    ("condition0", "condition4"),
    ("condition0", "condition5"),
    ("condition0", "condition6"),
]

TASK_IDS = ["streetlamp_to_cars", "past_pergolas", "above_building_europa"]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(
    successes: int, total: int, z: float = 1.96,
) -> tuple[float, float, float]:
    """Wilson score CI for a binomial proportion. Returns (rate, lower, upper)."""
    if total == 0:
        return (0.0, 0.0, 0.0)
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(
        p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)
    )
    return (p_hat, max(0.0, center - margin), min(1.0, center + margin))


def fishers_exact(
    a_succ: int, a_total: int, b_succ: int, b_total: int,
) -> dict[str, Any]:
    """Fisher's exact test on a 2x2 table. Returns p-value and odds ratio."""
    if not HAS_SCIPY:
        return {"p_value": None, "odds_ratio": None, "note": "scipy not installed"}
    table = [
        [a_succ, a_total - a_succ],
        [b_succ, b_total - b_succ],
    ]
    odds_ratio, p_value = scipy_stats.fisher_exact(table)
    return {
        "p_value": round(p_value, 6),
        "odds_ratio": round(odds_ratio, 4) if odds_ratio != float("inf") else "inf",
        "significant": p_value < 0.05,
    }


def mcnemars_exact(
    a_outcomes: list[int], b_outcomes: list[int],
) -> dict[str, Any]:
    """
    McNemar's exact test (binomial) for paired binary outcomes.
    a_outcomes and b_outcomes must be aligned by (task_id, variant).
    """
    if not HAS_SCIPY:
        return {"p_value": None, "note": "scipy not installed"}

    b_count = 0  # a=1, b=0
    c_count = 0  # a=0, b=1
    n_paired = 0
    for a, b in zip(a_outcomes, b_outcomes):
        if a is None or b is None:
            continue
        n_paired += 1
        if a == 1 and b == 0:
            b_count += 1
        elif a == 0 and b == 1:
            c_count += 1

    n_discordant = b_count + c_count
    if n_discordant == 0:
        return {
            "p_value": 1.0,
            "n_paired": n_paired,
            "n_discordant": 0,
            "significant": False,
            "note": "no discordant pairs",
        }

    p_value = scipy_stats.binomtest(b_count, n_discordant, 0.5).pvalue
    return {
        "p_value": round(p_value, 6),
        "n_paired": n_paired,
        "n_discordant": n_discordant,
        "b_count": b_count,
        "c_count": c_count,
        "significant": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotations(csv_path: str) -> list[dict]:
    """Load annotations CSV. Skips rows with empty task_success."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("task_success", "").strip()
            if ts == "":
                continue
            row["task_success"] = int(ts)
            sc = row.get("subgoals_completed", "").strip()
            st = row.get("subgoals_total", "").strip()
            row["subgoals_completed"] = int(sc) if sc else None
            row["subgoals_total"] = int(st) if st else None
            row["variant"] = int(row["variant"])
            rows.append(row)
    return rows


def load_diagnostic_metrics(results_dir: str) -> dict[str, list[dict]]:
    """Scan results_dir for run_info.json files, grouped by condition."""
    results_path = Path(results_dir)
    runs: dict[str, list[dict]] = defaultdict(list)
    for cond_dir in sorted(results_path.iterdir()):
        if not cond_dir.is_dir() or not cond_dir.name.startswith("condition"):
            continue
        for info_path in cond_dir.rglob("run_info.json"):
            try:
                with open(info_path) as f:
                    data = json.load(f)
                if data.get("aborted") or not data.get("completed", False):
                    continue
                runs[cond_dir.name].append(data)
            except (json.JSONDecodeError, OSError):
                pass
    return dict(runs)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_m1(rows: list[dict]) -> dict[str, dict]:
    """M1: Task success rate per condition."""
    by_cond: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r["task_success"])

    results = {}
    for cond in CONDITION_ORDER:
        outcomes = by_cond.get(cond, [])
        n = len(outcomes)
        s = sum(outcomes)
        rate, ci_lo, ci_hi = wilson_ci(s, n)
        results[cond] = {
            "successes": s,
            "total": n,
            "rate": round(rate, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        }
    return results


def compute_m3(rows: list[dict]) -> dict[str, dict]:
    """M3: Subgoal success rate per condition (pooled across episodes)."""
    by_cond: dict[str, dict] = defaultdict(lambda: {"completed": 0, "total": 0})
    for r in rows:
        if r["subgoals_completed"] is not None and r["subgoals_total"] is not None:
            by_cond[r["condition"]]["completed"] += r["subgoals_completed"]
            by_cond[r["condition"]]["total"] += r["subgoals_total"]

    results = {}
    for cond in CONDITION_ORDER:
        d = by_cond.get(cond, {"completed": 0, "total": 0})
        rate, ci_lo, ci_hi = wilson_ci(d["completed"], d["total"])
        results[cond] = {
            "completed": d["completed"],
            "total": d["total"],
            "rate": round(rate, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        }
    return results


def compute_per_task_breakdown(rows: list[dict]) -> dict[str, dict[str, list]]:
    """
    Per-task breakdown: for each condition and task, list variant outcomes.
    Returns {condition: {task_id: [variant1_result, variant2_result, variant3_result]}}.
    """
    breakdown: dict[str, dict[str, dict[int, int | None]]] = defaultdict(
        lambda: defaultdict(dict),
    )
    for r in rows:
        breakdown[r["condition"]][r["task_id"]][r["variant"]] = r["task_success"]

    result = {}
    for cond in CONDITION_ORDER:
        cond_data = {}
        for tid in TASK_IDS:
            variants = breakdown.get(cond, {}).get(tid, {})
            cond_data[tid] = [variants.get(v) for v in [1, 2, 3]]
        result[cond] = cond_data
    return result


def compute_task_level_pass(breakdown: dict[str, dict[str, list]]) -> dict[str, dict]:
    """
    Task-level pass rate: a task "passes" if majority of variants (2/3+) succeeded.
    Returns {condition: {tasks_passed, tasks_total, pass_rate}}.
    """
    results = {}
    for cond in CONDITION_ORDER:
        tasks = breakdown.get(cond, {})
        passed = 0
        total = 0
        for tid in TASK_IDS:
            variants = tasks.get(tid, [None, None, None])
            annotated = [v for v in variants if v is not None]
            if not annotated:
                continue
            total += 1
            if sum(annotated) >= 2:
                passed += 1
        results[cond] = {
            "tasks_passed": passed,
            "tasks_total": total,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
        }
    return results


def compute_statistical_tests(
    rows: list[dict], m1: dict[str, dict],
) -> dict[str, dict]:
    """Compute Fisher's exact and McNemar's exact for primary comparisons."""
    pair_key = lambda r: (r["task_id"], r["variant"])
    by_cond: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_cond[r["condition"]][pair_key(r)] = r["task_success"]

    results = {}
    for cond_a, cond_b in PRIMARY_COMPARISONS:
        label = f"{CONDITION_LABELS[cond_a].split(':')[0]} vs {CONDITION_LABELS[cond_b].split(':')[0]}"
        a_m1 = m1.get(cond_a, {})
        b_m1 = m1.get(cond_b, {})

        fisher = fishers_exact(
            a_m1.get("successes", 0), a_m1.get("total", 0),
            b_m1.get("successes", 0), b_m1.get("total", 0),
        )

        shared_keys = sorted(
            set(by_cond.get(cond_a, {}).keys())
            & set(by_cond.get(cond_b, {}).keys())
        )
        a_outcomes = [by_cond[cond_a][k] for k in shared_keys]
        b_outcomes = [by_cond[cond_b][k] for k in shared_keys]
        mcnemar = mcnemars_exact(a_outcomes, b_outcomes)

        results[label] = {
            "fisher": fisher,
            "mcnemar": mcnemar,
        }
    return results


def compute_diagnostic_summary(
    runs_by_cond: dict[str, list[dict]],
) -> dict[str, dict]:
    """Compute M4, M6, M7, M8 from run_info.json files."""
    m4_applicable = {"condition0", "condition2", "condition4", "condition5", "condition6"}
    results = {}

    for cond in CONDITION_ORDER:
        runs = runs_by_cond.get(cond, [])
        if not runs:
            continue
        n = len(runs)

        # M4: Correction rate
        m4 = None
        if cond in m4_applicable:
            total_conv = 0
            rescued = 0
            for run in runs:
                for sg in run.get("subgoal_summaries", []):
                    records = sg.get("vlm_call_records", [])
                    conv_count = sum(
                        1 for r in records
                        if r.get("label", "").startswith("convergence")
                    )
                    total_conv += conv_count
                    if sg.get("stop_reason") == "monitor_complete" and conv_count > 0:
                        rescued += max(0, conv_count - 1)
            if total_conv > 0:
                m4 = round(rescued / total_conv, 4)

        # M6: Latency
        all_rtts = []
        for run in runs:
            for r in run.get("vlm_call_records", []):
                if "rtt_s" in r:
                    all_rtts.append(r["rtt_s"])
        m6 = None
        if all_rtts:
            all_rtts.sort()
            nr = len(all_rtts)
            m6 = {
                "mean_rtt_s": round(sum(all_rtts) / nr, 3),
                "p95_rtt_s": round(all_rtts[min(int(nr * 0.95), nr - 1)], 3),
                "total_calls": nr,
            }

        # M7: Token overhead
        avg_input = sum(r.get("total_input_tokens", 0) for r in runs) / n
        avg_output = sum(r.get("total_output_tokens", 0) for r in runs) / n

        # M8: Episode length
        steps = [r.get("total_steps", 0) for r in runs]
        wall = [r.get("wall_clock_seconds", 0.0) for r in runs]

        results[cond] = {
            "n_runs": n,
            "M4_correction_rate": m4,
            "M6_latency": m6,
            "M7_avg_tokens_per_episode": round(avg_input + avg_output, 0),
            "M8_avg_steps": round(sum(steps) / n, 1),
            "M8_avg_wall_clock_s": round(sum(wall) / n, 1),
        }

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(
    m1: dict, m3: dict, breakdown: dict, task_pass: dict,
    stats: dict, diagnostics: dict | None,
):
    sep = "=" * 96
    print(sep)
    print(f"{'RVLN Experiment Results (Human-Annotated)':^96}")
    print(sep)
    print()

    # M1 Summary
    print("M1: Task Success Rate (human video review)")
    print(f"  {'Condition':<30} {'Success':>8} {'Rate':>8} {'95% Wilson CI':>22}")
    print("  " + "-" * 72)
    for cond in CONDITION_ORDER:
        d = m1.get(cond, {})
        if d.get("total", 0) == 0:
            continue
        label = CONDITION_LABELS.get(cond, cond)
        ci = f"[{d['ci_lower']:.3f}, {d['ci_upper']:.3f}]"
        print(f"  {label:<30} {d['successes']:>3}/{d['total']:<4} {d['rate']:>7.1%} {ci:>22}")
    print()

    # M3 Summary
    print("M3: Subgoal Success Rate (human video review)")
    print(f"  {'Condition':<30} {'Subgoals':>10} {'Rate':>8} {'95% Wilson CI':>22}")
    print("  " + "-" * 72)
    for cond in CONDITION_ORDER:
        d = m3.get(cond, {})
        if d.get("total", 0) == 0:
            continue
        label = CONDITION_LABELS.get(cond, cond)
        ci = f"[{d['ci_lower']:.3f}, {d['ci_upper']:.3f}]"
        print(f"  {label:<30} {d['completed']:>3}/{d['total']:<6} {d['rate']:>7.1%} {ci:>22}")
    print()

    # Per-task breakdown
    print("-" * 96)
    print("Per-Task Breakdown (v1/v2/v3 = variant outcomes, P = pass, F = fail, . = not annotated)")
    print()
    short_labels = {
        "condition0": "C0", "condition1": "C1", "condition2": "C2",
        "condition3": "C3", "condition4": "C4", "condition5": "C5",
        "condition6": "C6",
    }
    header = f"  {'Task':<28}"
    for cond in CONDITION_ORDER:
        header += f" {short_labels[cond]:>6}"
    print(header)
    print("  " + "-" * 72)

    for tid in TASK_IDS:
        row = f"  {tid:<28}"
        for cond in CONDITION_ORDER:
            variants = breakdown.get(cond, {}).get(tid, [None, None, None])
            syms = []
            for v in variants:
                if v is None:
                    syms.append(".")
                elif v == 1:
                    syms.append("P")
                else:
                    syms.append("F")
            row += f" {''.join(syms):>6}"
        print(row)
    print()

    # Task-level pass rate
    print("Task-Level Pass Rate (task passes if >= 2/3 variants succeed)")
    print(f"  {'Condition':<30} {'Passed':>8} {'Rate':>8}")
    print("  " + "-" * 50)
    for cond in CONDITION_ORDER:
        d = task_pass.get(cond, {})
        if d.get("tasks_total", 0) == 0:
            continue
        label = CONDITION_LABELS.get(cond, cond)
        print(f"  {label:<30} {d['tasks_passed']:>3}/{d['tasks_total']:<4} {d['pass_rate']:>7.1%}")
    print()

    # Statistical tests
    if stats:
        print("-" * 96)
        print("Primary Comparisons (pre-specified, unadjusted alpha = 0.05)")
        print(f"  {'Comparison':<24} {'Test':<12} {'p-value':>10} {'Effect':>10} {'Sig?':>6}")
        print("  " + "-" * 66)
        for label, tests in stats.items():
            f = tests.get("fisher", {})
            m = tests.get("mcnemar", {})
            f_p = f.get("p_value")
            f_or = f.get("odds_ratio")
            f_sig = f.get("significant")
            m_p = m.get("p_value")
            m_sig = m.get("significant")
            m_disc = m.get("n_discordant", "")

            f_p_str = f"{f_p:.4f}" if f_p is not None else "N/A"
            f_or_str = f"{f_or:.2f}" if isinstance(f_or, (int, float)) else str(f_or or "N/A")
            f_sig_str = "  *" if f_sig else ""
            print(f"  {label:<24} {'Fisher':.<12} {f_p_str:>10} {f_or_str:>10}{f_sig_str:>6}")

            m_p_str = f"{m_p:.4f}" if m_p is not None else "N/A"
            m_note = m.get("note", "")
            m_eff = f"d={m_disc}" if m_disc != "" else ""
            m_sig_str = "  *" if m_sig else ""
            if m_note:
                m_eff = m_note[:10]
            print(f"  {'':<24} {'McNemar':.<12} {m_p_str:>10} {m_eff:>10}{m_sig_str:>6}")
        print()
        print("  * = significant at p < 0.05 (unadjusted)")
        print()

    # Secondary comparisons (descriptive)
    print("-" * 96)
    print("Secondary Comparisons (descriptive only, no hypothesis tests)")
    for cond_a, cond_b in SECONDARY_COMPARISONS:
        a_d = m1.get(cond_a, {})
        b_d = m1.get(cond_b, {})
        a_label = CONDITION_LABELS.get(cond_a, cond_a).split(":")[0]
        b_label = CONDITION_LABELS.get(cond_b, cond_b)
        a_str = f"{a_d.get('successes', 0)}/{a_d.get('total', 0)}"
        b_str = f"{b_d.get('successes', 0)}/{b_d.get('total', 0)}"
        diff = (a_d.get("rate", 0) - b_d.get("rate", 0))
        print(f"  {a_label} vs {b_label}: {a_str} vs {b_str} (diff = {diff:+.1%})")
    print()

    # Diagnostics
    if diagnostics:
        print("-" * 96)
        print("Diagnostic Metrics (from run_info.json)")
        print(f"  {'Condition':<30} {'M4(Corr)':>9} {'M6(RTT)':>9} {'M7(Tok)':>10} {'M8(Steps)':>10} {'M8(sec)':>9}")
        print("  " + "-" * 80)
        for cond in CONDITION_ORDER:
            d = diagnostics.get(cond)
            if not d:
                continue
            label = CONDITION_LABELS.get(cond, cond)
            m4_str = f"{d['M4_correction_rate']:.1%}" if d["M4_correction_rate"] is not None else "N/A"
            m6_str = f"{d['M6_latency']['mean_rtt_s']:.2f}s" if d["M6_latency"] else "N/A"
            m7_str = f"{d['M7_avg_tokens_per_episode']:.0f}"
            m8s_str = f"{d['M8_avg_steps']:.0f}"
            m8w_str = f"{d['M8_avg_wall_clock_s']:.0f}"
            print(f"  {label:<30} {m4_str:>9} {m6_str:>9} {m7_str:>10} {m8s_str:>10} {m8w_str:>9}")
        print()

    # Notes
    print(sep)
    print("Notes:")
    print("  - All task success (M1) and subgoal success (M3) determined by human video review.")
    print("  - Task-level pass: task passes if >= 2/3 variants succeed (effective n=3).")
    print("  - Primary comparisons: C0 vs C3 (monitoring), C0 vs C1 (decomposition).")
    print("  - Secondary comparisons: descriptive only (proportions + CIs).")
    print("  - Episode-level n=9 (3 tasks x 3 variants). Within-task variants are")
    print("    correlated (small perturbations), so effective independent n ~ 3.")
    if not HAS_SCIPY:
        print("  - [WARNING] scipy not installed. Statistical tests were skipped.")
    print(sep)


def write_json_output(
    m1: dict, m3: dict, breakdown: dict, task_pass: dict,
    stats: dict, diagnostics: dict | None, output_path: Path,
):
    output = {
        "metadata": {
            "script": "analyze_annotations.py",
            "success_criterion": "human video review (all conditions)",
            "primary_comparisons": ["C0 vs C3", "C0 vs C1"],
            "alpha": 0.05,
            "correction": "none (pre-specified comparisons, n=9)",
            "scipy_available": HAS_SCIPY,
        },
        "M1_task_success": m1,
        "M3_subgoal_success": m3,
        "per_task_breakdown": breakdown,
        "task_level_pass": task_pass,
        "primary_statistical_tests": stats,
    }
    if diagnostics:
        output["diagnostic_metrics"] = diagnostics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON output: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze human-annotated RVLN experimental results.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="results/annotations.csv",
        help="Path to annotations CSV (default: results/annotations.csv)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to results directory for diagnostic metrics (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/analysis_output.json",
        help="Path for JSON output (default: results/analysis_output.json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.annotations):
        print(f"Annotations file not found: {args.annotations}", file=sys.stderr)
        print("Fill in results/annotations.csv first.", file=sys.stderr)
        sys.exit(1)

    rows = load_annotations(args.annotations)
    if not rows:
        print("No annotated rows found. Fill in task_success column.", file=sys.stderr)
        sys.exit(1)

    annotated_conds = set(r["condition"] for r in rows)
    missing = [c for c in CONDITION_ORDER if c not in annotated_conds]
    if missing:
        print(f"Note: no annotations for {', '.join(missing)}", file=sys.stderr)

    print(f"Loaded {len(rows)} annotated episodes from {args.annotations}")
    for cond in CONDITION_ORDER:
        n = sum(1 for r in rows if r["condition"] == cond)
        if n > 0:
            print(f"  {CONDITION_LABELS.get(cond, cond)}: {n} episodes")
    print()

    m1 = compute_m1(rows)
    m3 = compute_m3(rows)
    breakdown = compute_per_task_breakdown(rows)
    task_pass = compute_task_level_pass(breakdown)
    stats = compute_statistical_tests(rows, m1)

    diagnostics = None
    if args.results_dir and os.path.isdir(args.results_dir):
        diag_runs = load_diagnostic_metrics(args.results_dir)
        if diag_runs:
            diagnostics = compute_diagnostic_summary(diag_runs)

    print_results(m1, m3, breakdown, task_pass, stats, diagnostics)
    write_json_output(m1, m3, breakdown, task_pass, stats, diagnostics, Path(args.output))


if __name__ == "__main__":
    main()
