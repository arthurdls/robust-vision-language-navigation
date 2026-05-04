#!/usr/bin/env python3
"""
Aggregate experimental results across all RVLN conditions and compute metrics.

Experimental design: 7 conditions (C0-C6), 3 maps, 5 tasks per map,
3 starting position variations per task = 45 episodes per condition, 315 total.

Metrics computed:
  Primary:
    M1 - Task Success Rate (all subgoals achieved)
    M2 - Constraint Adherence Rate (no constraints violated)
    M3 - Subgoal Success Rate (fraction of individual subgoals completed)
  Diagnostic:
    M4 - Supervisor Correction Rate (rescued / total premature convergences)
    M5 - (Manual annotation required, not computed here)
    M6 - Control Frequency / Latency (VLM call RTT statistics)
    M7 - Token / Inference Overhead (input + output tokens per episode)
    M8 - Average Episode Length (steps and wall-clock seconds)

Breakdowns:
  - Per-category (sequential vs constrained tasks)
  - Per-map (generalization across simulation environments)

Statistical tests:
  - Wilson confidence intervals for binary metrics (M1, M2, M3)
  - Fisher's exact test for pairwise C0 vs each baseline
  - McNemar's test for paired comparisons (matched starting positions)
  - Bonferroni correction for 6 comparisons (threshold: p < 0.0083)
  - Effect size (odds ratio) reporting
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITION_NAMES = {
    "condition0": "C0: Full System",
    "condition1": "C1: Naive (no subgoals)",
    "condition2": "C2: LLM Planner Only",
    "condition3": "C3: Open Loop",
    "condition4": "C4: Single Frame",
    "condition5": "C5: Grid Only",
    "condition6": "C6: Text-Only Global",
}

CONDITION_ORDER = [
    "condition0", "condition1", "condition2", "condition3",
    "condition4", "condition5", "condition6",
]

# Bonferroni-corrected significance threshold (0.05 / 6 comparisons)
BONFERRONI_THRESHOLD = 0.05 / 6  # ~0.0083


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.
    Returns (point_estimate, lower_bound, upper_bound).
    """
    if total == 0:
        return (0.0, 0.0, 0.0)
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (p_hat, lower, upper)


def fishers_exact_test(
    c0_successes: int, c0_total: int,
    cx_successes: int, cx_total: int,
) -> dict[str, Any]:
    """
    Run Fisher's exact test comparing C0 to another condition.
    Returns dict with p_value, odds_ratio, and significance flag.
    """
    if not HAS_SCIPY:
        return {
            "p_value": None,
            "odds_ratio": None,
            "significant_bonferroni": None,
            "note": "scipy not installed",
        }
    # Construct 2x2 contingency table
    # [[c0_success, c0_fail], [cx_success, cx_fail]]
    table = [
        [c0_successes, c0_total - c0_successes],
        [cx_successes, cx_total - cx_successes],
    ]
    odds_ratio, p_value = scipy_stats.fisher_exact(table)
    return {
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "significant_bonferroni": p_value < BONFERRONI_THRESHOLD,
    }


def mcnemars_test(
    c0_outcomes: list[bool | None],
    cx_outcomes: list[bool | None],
) -> dict[str, Any]:
    """
    McNemar's test for paired binary outcomes (matched starting positions).

    Each list entry corresponds to the same episode (same task + same starting
    position) run under two conditions.  Entries that are None (manual review
    needed) are excluded from the test.

    Returns dict with statistic, p_value, n_discordant, and significance flag.
    """
    if not HAS_SCIPY:
        return {
            "statistic": None,
            "p_value": None,
            "n_discordant": None,
            "significant_bonferroni": None,
            "note": "scipy not installed",
        }

    # Build discordant pair counts
    b = 0  # c0 success, cx failure
    c = 0  # c0 failure, cx success
    n_paired = 0
    for o0, ox in zip(c0_outcomes, cx_outcomes):
        if o0 is None or ox is None:
            continue
        n_paired += 1
        if o0 and not ox:
            b += 1
        elif not o0 and ox:
            c += 1

    n_discordant = b + c
    if n_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_paired": n_paired,
            "n_discordant": 0,
            "significant_bonferroni": False,
            "note": "no discordant pairs",
        }

    if n_discordant < 25:
        p_value = scipy_stats.binomtest(b, n_discordant, 0.5).pvalue
        return {
            "statistic": None,
            "p_value": p_value,
            "n_paired": n_paired,
            "n_discordant": n_discordant,
            "significant_bonferroni": p_value < BONFERRONI_THRESHOLD,
            "note": "exact binomial (n_discordant < 25)",
        }

    statistic = (b - c) ** 2 / (b + c)
    p_value = 1 - scipy_stats.chi2.cdf(statistic, df=1)
    return {
        "statistic": round(statistic, 4),
        "p_value": p_value,
        "n_paired": n_paired,
        "n_discordant": n_discordant,
        "significant_bonferroni": p_value < BONFERRONI_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_run_infos(results_dir: str, conditions: list[str] | None = None) -> dict[str, list[dict]]:
    """
    Scan results directory and load all run_info.json files, grouped by condition.

    Handles two directory structures:
      - results/conditionN/<map_dir>/<run_name>/run_info.json  (multi-map)
      - results/conditionN/<run_name>/run_info.json            (legacy/flat)
    """
    results_path = Path(results_dir)
    runs_by_condition: dict[str, list[dict]] = defaultdict(list)

    for cond_dir in sorted(results_path.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond_name = cond_dir.name
        # Skip non-condition directories (like "old")
        if not cond_name.startswith("condition"):
            continue
        if conditions and cond_name not in conditions:
            continue

        # Walk to find run_info.json files
        for run_info_path in cond_dir.rglob("run_info.json"):
            try:
                with open(run_info_path, "r") as f:
                    data = json.load(f)
                # Attach the file path for reference
                data["_source_path"] = str(run_info_path)
                # Check for companion constraint_analysis.json
                constraint_path = run_info_path.parent / "constraint_analysis.json"
                if constraint_path.exists():
                    with open(constraint_path, "r") as f:
                        data["_constraint_analysis"] = json.load(f)
                runs_by_condition[cond_name].append(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [WARN] Could not load {run_info_path}: {e}", file=sys.stderr)

    return dict(runs_by_condition)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def is_task_success(run: dict) -> bool | None:
    """
    Determine if an episode was a task success.

    For conditions with subgoals: all subgoals must have stop_reason == "monitor_complete".
    For C1 (naive, no subgoals): returns None (requires manual annotation).
    For C3 (open loop): returns None (convergence does not prove goal
      achievement; requires manual video review).
    """
    condition = run.get("condition", "")
    subgoal_summaries = run.get("subgoal_summaries")

    # C1 naive: no subgoals, no monitor
    if "condition1" in condition or "naive" in condition:
        ca = run.get("_constraint_analysis")
        if ca and "task_success" in ca:
            return bool(ca["task_success"])
        return None

    # C3 open-loop: convergence is self-fulfilling, not a real success signal
    if "condition3" in condition or "open_loop" in condition:
        ca = run.get("_constraint_analysis")
        if ca and "task_success" in ca:
            return bool(ca["task_success"])
        return None

    # Conditions with subgoal_summaries
    if subgoal_summaries:
        for sg in subgoal_summaries:
            stop = sg.get("stop_reason", "")
            if stop != "monitor_complete":
                return False
        return True

    # Fallback: single-episode conditions without subgoals
    stop_reason = run.get("stop_reason", "")
    if stop_reason == "monitor_complete":
        return True
    return False


def is_constraint_adhered(run: dict) -> bool | None:
    """
    Check constraint adherence for an episode.
    Uses constraint_analysis.json if available, otherwise falls back to
    any_constraint_violated field in run_info.
    """
    ca = run.get("_constraint_analysis")
    if ca:
        if "any_constraint_violated" in ca:
            return not ca["any_constraint_violated"]
        if "constraint_adherence" in ca:
            return bool(ca["constraint_adherence"])

    if "any_constraint_violated" in run:
        return not run["any_constraint_violated"]

    # Check subgoal-level constraint violations
    subgoal_summaries = run.get("subgoal_summaries", [])
    if subgoal_summaries:
        for sg in subgoal_summaries:
            if sg.get("constraint_violation_count", 0) > 0:
                return False
        return True

    # Cannot determine
    return None


def compute_subgoal_success_rate(run: dict) -> float | None:
    """
    Compute fraction of subgoals that were successfully completed in an episode.
    Returns None for C1 (no subgoals) and C3 (convergence is not a real signal).
    """
    condition = run.get("condition", "")
    if "condition1" in condition or "naive" in condition:
        return None
    if "condition3" in condition or "open_loop" in condition:
        return None

    subgoal_summaries = run.get("subgoal_summaries", [])
    if not subgoal_summaries:
        return None

    successes = 0
    for sg in subgoal_summaries:
        stop = sg.get("stop_reason", "")
        if stop == "monitor_complete":
            successes += 1

    return successes / len(subgoal_summaries)


def compute_correction_rate(run: dict) -> dict[str, int] | None:
    """
    Compute supervisor correction metrics.
    Returns dict with total_convergences, rescued_convergences, corrections_used.
    Rescued = convergence events that did NOT end the subgoal (i.e., supervisor intervened).
    """
    subgoal_summaries = run.get("subgoal_summaries", [])
    if not subgoal_summaries:
        return None

    total_convergences = 0
    rescued_convergences = 0
    total_corrections = 0

    for sg in subgoal_summaries:
        corrections = sg.get("corrections_used", 0)
        total_corrections += corrections
        # Count convergence events in vlm_call_records
        records = sg.get("vlm_call_records", [])
        convergence_count = sum(1 for r in records if r.get("label", "").startswith("convergence"))
        total_convergences += convergence_count
        # If the subgoal eventually completed (monitor_complete) but had convergences,
        # those early convergences were "rescued" by the supervisor
        stop = sg.get("stop_reason", "")
        if stop == "monitor_complete" and convergence_count > 0:
            # The final convergence led to completion; earlier ones were rescued
            rescued_convergences += max(0, convergence_count - 1)
        elif stop != "monitor_complete" and stop != "convergence":
            # All convergences were rescued (task ended for another reason)
            rescued_convergences += convergence_count
        elif stop == "max_steps" and convergence_count > 0:
            # All convergences were rescued but task still timed out
            rescued_convergences += convergence_count

    return {
        "total_convergences": total_convergences,
        "rescued_convergences": rescued_convergences,
        "total_corrections": total_corrections,
    }


def compute_latency_stats(run: dict) -> dict[str, float] | None:
    """
    Compute VLM call latency statistics from vlm_call_records.
    Returns dict with mean_rtt, median_rtt, p95_rtt, total_calls.
    """
    records = run.get("vlm_call_records", [])
    if not records:
        return None

    rtts = [r["rtt_s"] for r in records if "rtt_s" in r]
    if not rtts:
        return None

    rtts_sorted = sorted(rtts)
    n = len(rtts_sorted)
    mean_rtt = sum(rtts) / n
    median_rtt = rtts_sorted[n // 2] if n % 2 == 1 else (rtts_sorted[n // 2 - 1] + rtts_sorted[n // 2]) / 2
    p95_idx = min(int(n * 0.95), n - 1)
    p95_rtt = rtts_sorted[p95_idx]

    return {
        "mean_rtt_s": round(mean_rtt, 3),
        "median_rtt_s": round(median_rtt, 3),
        "p95_rtt_s": round(p95_rtt, 3),
        "total_calls": n,
    }


def compute_token_overhead(run: dict) -> dict[str, int]:
    """Compute token usage for an episode."""
    return {
        "input_tokens": run.get("total_input_tokens", 0),
        "output_tokens": run.get("total_output_tokens", 0),
        "total_tokens": run.get("total_input_tokens", 0) + run.get("total_output_tokens", 0),
    }


def compute_episode_length(run: dict) -> dict[str, float]:
    """Compute episode length in steps and wall-clock time."""
    return {
        "total_steps": run.get("total_steps", 0),
        "wall_clock_seconds": run.get("wall_clock_seconds", 0.0),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_condition(condition: str, runs: list[dict]) -> dict[str, Any]:
    """Compute all metrics for a single condition."""
    n = len(runs)
    if n == 0:
        return {"n_episodes": 0, "error": "no runs found"}

    # M1: Task Success Rate
    task_successes = 0
    task_determined = 0
    task_requires_manual = 0
    for run in runs:
        result = is_task_success(run)
        if result is None:
            task_requires_manual += 1
        else:
            task_determined += 1
            if result:
                task_successes += 1

    m1_rate, m1_ci_low, m1_ci_high = wilson_ci(task_successes, task_determined)

    # M2: Constraint Adherence Rate
    constraint_adhered = 0
    constraint_determined = 0
    for run in runs:
        result = is_constraint_adhered(run)
        if result is not None:
            constraint_determined += 1
            if result:
                constraint_adhered += 1

    m2_rate, m2_ci_low, m2_ci_high = wilson_ci(constraint_adhered, constraint_determined)

    # M3: Subgoal Success Rate
    subgoal_rates = []
    total_subgoals = 0
    successful_subgoals = 0
    for run in runs:
        cond = run.get("condition", "")
        if "condition3" in cond or "open_loop" in cond:
            continue
        summaries = run.get("subgoal_summaries", [])
        if not summaries:
            continue
        for sg in summaries:
            total_subgoals += 1
            stop = sg.get("stop_reason", "")
            if stop == "monitor_complete":
                successful_subgoals += 1
        rate = compute_subgoal_success_rate(run)
        if rate is not None:
            subgoal_rates.append(rate)

    m3_rate, m3_ci_low, m3_ci_high = wilson_ci(successful_subgoals, total_subgoals)

    # M4: Supervisor Correction Rate
    total_convergences = 0
    total_rescued = 0
    total_corrections = 0
    for run in runs:
        cr = compute_correction_rate(run)
        if cr:
            total_convergences += cr["total_convergences"]
            total_rescued += cr["rescued_convergences"]
            total_corrections += cr["total_corrections"]

    m4_rate = total_rescued / total_convergences if total_convergences > 0 else None

    # M6: Latency
    all_rtts = []
    for run in runs:
        records = run.get("vlm_call_records", [])
        for r in records:
            if "rtt_s" in r:
                all_rtts.append(r["rtt_s"])

    if all_rtts:
        all_rtts_sorted = sorted(all_rtts)
        n_rtts = len(all_rtts_sorted)
        m6_mean = sum(all_rtts) / n_rtts
        m6_median = (
            all_rtts_sorted[n_rtts // 2]
            if n_rtts % 2 == 1
            else (all_rtts_sorted[n_rtts // 2 - 1] + all_rtts_sorted[n_rtts // 2]) / 2
        )
        m6_p95 = all_rtts_sorted[min(int(n_rtts * 0.95), n_rtts - 1)]
    else:
        m6_mean = m6_median = m6_p95 = None

    # M7: Token overhead
    total_input = sum(run.get("total_input_tokens", 0) for run in runs)
    total_output = sum(run.get("total_output_tokens", 0) for run in runs)
    avg_input = total_input / n
    avg_output = total_output / n

    # M8: Episode length
    all_steps = [run.get("total_steps", 0) for run in runs]
    all_wall = [run.get("wall_clock_seconds", 0.0) for run in runs]
    avg_steps = sum(all_steps) / n
    avg_wall = sum(all_wall) / n

    return {
        "n_episodes": n,
        "M1_task_success": {
            "successes": task_successes,
            "determined": task_determined,
            "requires_manual": task_requires_manual,
            "rate": round(m1_rate, 4),
            "wilson_ci_lower": round(m1_ci_low, 4),
            "wilson_ci_upper": round(m1_ci_high, 4),
        },
        "M2_constraint_adherence": {
            "adhered": constraint_adhered,
            "determined": constraint_determined,
            "rate": round(m2_rate, 4),
            "wilson_ci_lower": round(m2_ci_low, 4),
            "wilson_ci_upper": round(m2_ci_high, 4),
        },
        "M3_subgoal_success": {
            "successful_subgoals": successful_subgoals,
            "total_subgoals": total_subgoals,
            "rate": round(m3_rate, 4),
            "wilson_ci_lower": round(m3_ci_low, 4),
            "wilson_ci_upper": round(m3_ci_high, 4),
            "mean_per_episode": round(sum(subgoal_rates) / len(subgoal_rates), 4) if subgoal_rates else None,
        },
        "M4_correction_rate": {
            "total_premature_convergences": total_convergences,
            "rescued_convergences": total_rescued,
            "rate": round(m4_rate, 4) if m4_rate is not None else None,
            "total_corrections_issued": total_corrections,
        },
        "M5_qualitative": {
            "note": "Requires manual video annotation. Not computed automatically.",
        },
        "M6_latency": {
            "mean_rtt_s": round(m6_mean, 3) if m6_mean is not None else None,
            "median_rtt_s": round(m6_median, 3) if m6_median is not None else None,
            "p95_rtt_s": round(m6_p95, 3) if m6_p95 is not None else None,
            "total_vlm_calls": len(all_rtts),
        },
        "M7_token_overhead": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_input_tokens_per_episode": round(avg_input, 1),
            "avg_output_tokens_per_episode": round(avg_output, 1),
            "avg_total_tokens_per_episode": round(avg_input + avg_output, 1),
        },
        "M8_episode_length": {
            "avg_steps": round(avg_steps, 1),
            "avg_wall_clock_s": round(avg_wall, 1),
            "min_steps": min(all_steps) if all_steps else 0,
            "max_steps": max(all_steps) if all_steps else 0,
            "min_wall_clock_s": round(min(all_wall), 1) if all_wall else 0.0,
            "max_wall_clock_s": round(max(all_wall), 1) if all_wall else 0.0,
        },
    }


def aggregate_by_category(runs_by_condition: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    """
    Group episodes by task category and compute M1/M2/M3 for each category
    within each condition.

    Returns a nested dict: {condition: {category: {M1, M2, M3 metrics}}}.
    Episodes whose task lacks a category field are grouped under "unknown".
    """
    result: dict[str, dict[str, dict]] = {}

    for cond in CONDITION_ORDER:
        runs = runs_by_condition.get(cond, [])
        if not runs:
            continue

        # Group runs by category
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for run in runs:
            task_info = run.get("task", {})
            category = task_info.get("category", "unknown")
            by_cat[category].append(run)

        cond_cats = {}
        for cat, cat_runs in sorted(by_cat.items()):
            n = len(cat_runs)

            # M1: Task Success Rate
            task_successes = 0
            task_determined = 0
            for run in cat_runs:
                res = is_task_success(run)
                if res is not None:
                    task_determined += 1
                    if res:
                        task_successes += 1
            m1_rate, m1_lo, m1_hi = wilson_ci(task_successes, task_determined)

            # M2: Constraint Adherence Rate
            constraint_adhered = 0
            constraint_determined = 0
            for run in cat_runs:
                res = is_constraint_adhered(run)
                if res is not None:
                    constraint_determined += 1
                    if res:
                        constraint_adhered += 1
            m2_rate, m2_lo, m2_hi = wilson_ci(constraint_adhered, constraint_determined)

            # M3: Subgoal Success Rate
            successful_subgoals = 0
            total_subgoals = 0
            for run in cat_runs:
                c = run.get("condition", "")
                if "condition3" in c or "open_loop" in c:
                    continue
                summaries = run.get("subgoal_summaries", [])
                for sg in summaries:
                    total_subgoals += 1
                    stop = sg.get("stop_reason", "")
                    if stop == "monitor_complete":
                        successful_subgoals += 1
            m3_rate, m3_lo, m3_hi = wilson_ci(successful_subgoals, total_subgoals)

            cond_cats[cat] = {
                "n_episodes": n,
                "M1_task_success": {
                    "successes": task_successes,
                    "determined": task_determined,
                    "rate": round(m1_rate, 4),
                    "wilson_ci_lower": round(m1_lo, 4),
                    "wilson_ci_upper": round(m1_hi, 4),
                },
                "M2_constraint_adherence": {
                    "adhered": constraint_adhered,
                    "determined": constraint_determined,
                    "rate": round(m2_rate, 4),
                    "wilson_ci_lower": round(m2_lo, 4),
                    "wilson_ci_upper": round(m2_hi, 4),
                },
                "M3_subgoal_success": {
                    "successful_subgoals": successful_subgoals,
                    "total_subgoals": total_subgoals,
                    "rate": round(m3_rate, 4),
                    "wilson_ci_lower": round(m3_lo, 4),
                    "wilson_ci_upper": round(m3_hi, 4),
                },
            }

        result[cond] = cond_cats

    return result


def aggregate_by_map(runs_by_condition: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    """
    Group episodes by map and compute M1/M2/M3 for each map within each
    condition.  Map is inferred from the run's source path (e.g.
    results/condition0/greek_island/...) or the run's env_id field.

    Returns: {condition: {map_name: {M1, M2, M3 metrics}}}.
    """
    result: dict[str, dict[str, dict]] = {}

    for cond in CONDITION_ORDER:
        runs = runs_by_condition.get(cond, [])
        if not runs:
            continue

        by_map: dict[str, list[dict]] = defaultdict(list)
        for run in runs:
            map_name = run.get("env_id", "")
            if not map_name:
                source = run.get("_source_path", "")
                parts = Path(source).parts
                for i, p in enumerate(parts):
                    if p.startswith("condition"):
                        if i + 1 < len(parts) and not parts[i + 1].startswith("c"):
                            map_name = parts[i + 1]
                        break
            if not map_name:
                map_name = "unknown"
            by_map[map_name].append(run)

        cond_maps = {}
        for map_name, map_runs in sorted(by_map.items()):
            n = len(map_runs)

            task_successes = 0
            task_determined = 0
            for run in map_runs:
                res = is_task_success(run)
                if res is not None:
                    task_determined += 1
                    if res:
                        task_successes += 1
            m1_rate, m1_lo, m1_hi = wilson_ci(task_successes, task_determined)

            constraint_adhered = 0
            constraint_determined = 0
            for run in map_runs:
                res = is_constraint_adhered(run)
                if res is not None:
                    constraint_determined += 1
                    if res:
                        constraint_adhered += 1
            m2_rate, m2_lo, m2_hi = wilson_ci(constraint_adhered, constraint_determined)

            successful_subgoals = 0
            total_subgoals = 0
            for run in map_runs:
                c = run.get("condition", "")
                if "condition3" in c or "open_loop" in c:
                    continue
                summaries = run.get("subgoal_summaries", [])
                for sg in summaries:
                    total_subgoals += 1
                    if sg.get("stop_reason", "") == "monitor_complete":
                        successful_subgoals += 1
            m3_rate, m3_lo, m3_hi = wilson_ci(successful_subgoals, total_subgoals)

            cond_maps[map_name] = {
                "n_episodes": n,
                "M1_task_success": {
                    "successes": task_successes,
                    "determined": task_determined,
                    "rate": round(m1_rate, 4),
                    "wilson_ci_lower": round(m1_lo, 4),
                    "wilson_ci_upper": round(m1_hi, 4),
                },
                "M2_constraint_adherence": {
                    "adhered": constraint_adhered,
                    "determined": constraint_determined,
                    "rate": round(m2_rate, 4),
                    "wilson_ci_lower": round(m2_lo, 4),
                    "wilson_ci_upper": round(m2_hi, 4),
                },
                "M3_subgoal_success": {
                    "successful_subgoals": successful_subgoals,
                    "total_subgoals": total_subgoals,
                    "rate": round(m3_rate, 4),
                    "wilson_ci_lower": round(m3_lo, 4),
                    "wilson_ci_upper": round(m3_hi, 4),
                },
            }

        result[cond] = cond_maps

    return result


def compute_paired_mcnemar_tests(
    runs_by_condition: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    Compute McNemar's test for C0 vs each baseline on M1, using paired
    episodes matched by task_id.

    Starting positions are fixed across conditions, so episodes with the
    same task_id form a natural pair.
    """
    comparisons = {}
    c0_runs = runs_by_condition.get("condition0", [])
    if not c0_runs:
        return comparisons

    c0_by_task = {}
    for run in c0_runs:
        tid = run.get("task", {}).get("task_id", "")
        if tid:
            c0_by_task[tid] = run

    for cond in CONDITION_ORDER[1:]:
        cx_runs = runs_by_condition.get(cond, [])
        if not cx_runs:
            continue

        cx_by_task = {}
        for run in cx_runs:
            tid = run.get("task", {}).get("task_id", "")
            if tid:
                cx_by_task[tid] = run

        shared_tasks = sorted(set(c0_by_task.keys()) & set(cx_by_task.keys()))
        if not shared_tasks:
            continue

        c0_outcomes = [is_task_success(c0_by_task[t]) for t in shared_tasks]
        cx_outcomes = [is_task_success(cx_by_task[t]) for t in shared_tasks]

        comparisons[f"C0_vs_{cond}"] = {
            "M1_task_success_mcnemar": mcnemars_test(c0_outcomes, cx_outcomes),
        }

    return comparisons


def compute_pairwise_tests(
    condition_metrics: dict[str, dict],
) -> dict[str, dict]:
    """
    Compute Fisher's exact test for C0 vs each other condition,
    for M1, M2, and M3.
    """
    comparisons = {}
    c0 = condition_metrics.get("condition0")
    if not c0:
        return comparisons

    for cond in CONDITION_ORDER[1:]:
        cx = condition_metrics.get(cond)
        if not cx:
            continue

        cond_comparisons = {}

        # M1
        c0_m1 = c0["M1_task_success"]
        cx_m1 = cx["M1_task_success"]
        if c0_m1["determined"] > 0 and cx_m1["determined"] > 0:
            cond_comparisons["M1_task_success"] = fishers_exact_test(
                c0_m1["successes"], c0_m1["determined"],
                cx_m1["successes"], cx_m1["determined"],
            )

        # M2
        c0_m2 = c0["M2_constraint_adherence"]
        cx_m2 = cx["M2_constraint_adherence"]
        if c0_m2["determined"] > 0 and cx_m2["determined"] > 0:
            cond_comparisons["M2_constraint_adherence"] = fishers_exact_test(
                c0_m2["adhered"], c0_m2["determined"],
                cx_m2["adhered"], cx_m2["determined"],
            )

        # M3
        c0_m3 = c0["M3_subgoal_success"]
        cx_m3 = cx["M3_subgoal_success"]
        if c0_m3["total_subgoals"] > 0 and cx_m3["total_subgoals"] > 0:
            cond_comparisons["M3_subgoal_success"] = fishers_exact_test(
                c0_m3["successful_subgoals"], c0_m3["total_subgoals"],
                cx_m3["successful_subgoals"], cx_m3["total_subgoals"],
            )

        comparisons[f"C0_vs_{cond}"] = cond_comparisons

    return comparisons


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary_table(condition_metrics: dict[str, dict], pairwise: dict[str, dict]):
    """Print a formatted summary table to stdout."""
    sep = "=" * 100
    print(sep)
    print(f"{'RVLN Experiment Results Summary':^100}")
    print(sep)
    print()

    # Header
    header = f"{'Condition':<28} {'N':>4} {'M1 (Task)':>12} {'M2 (Const)':>12} {'M3 (Subg)':>12} {'M8 (Steps)':>10} {'M8 (sec)':>9}"
    print(header)
    print("-" * 100)

    for cond in CONDITION_ORDER:
        metrics = condition_metrics.get(cond)
        if not metrics:
            continue
        name = CONDITION_NAMES.get(cond, cond)
        n = metrics["n_episodes"]
        m1 = metrics["M1_task_success"]
        m2 = metrics["M2_constraint_adherence"]
        m3 = metrics["M3_subgoal_success"]
        m8 = metrics["M8_episode_length"]

        m1_str = f"{m1['rate']:.2%}" if m1["determined"] > 0 else "N/A"
        m2_str = f"{m2['rate']:.2%}" if m2["determined"] > 0 else "N/A"
        m3_str = f"{m3['rate']:.2%}" if m3["total_subgoals"] > 0 else "N/A"

        print(
            f"{name:<28} {n:>4} {m1_str:>12} {m2_str:>12} {m3_str:>12} "
            f"{m8['avg_steps']:>10.1f} {m8['avg_wall_clock_s']:>9.1f}"
        )

    print()
    print("-" * 100)
    print()

    # Confidence intervals
    print("Wilson 95% Confidence Intervals:")
    print(f"{'Condition':<28} {'M1 CI':>20} {'M2 CI':>20} {'M3 CI':>20}")
    print("-" * 100)
    for cond in CONDITION_ORDER:
        metrics = condition_metrics.get(cond)
        if not metrics:
            continue
        name = CONDITION_NAMES.get(cond, cond)
        m1 = metrics["M1_task_success"]
        m2 = metrics["M2_constraint_adherence"]
        m3 = metrics["M3_subgoal_success"]

        m1_ci = f"[{m1['wilson_ci_lower']:.3f}, {m1['wilson_ci_upper']:.3f}]" if m1["determined"] > 0 else "N/A"
        m2_ci = f"[{m2['wilson_ci_lower']:.3f}, {m2['wilson_ci_upper']:.3f}]" if m2["determined"] > 0 else "N/A"
        m3_ci = f"[{m3['wilson_ci_lower']:.3f}, {m3['wilson_ci_upper']:.3f}]" if m3["total_subgoals"] > 0 else "N/A"

        print(f"{name:<28} {m1_ci:>20} {m2_ci:>20} {m3_ci:>20}")

    print()

    # Diagnostic metrics
    print("-" * 100)
    print("Diagnostic Metrics:")
    print(f"{'Condition':<28} {'M4 (Corr%)':>10} {'M6 (RTT)':>10} {'M7 (Tok/ep)':>12} {'VLM Calls':>10}")
    print("-" * 100)
    for cond in CONDITION_ORDER:
        metrics = condition_metrics.get(cond)
        if not metrics:
            continue
        name = CONDITION_NAMES.get(cond, cond)
        m4 = metrics["M4_correction_rate"]
        m6 = metrics["M6_latency"]
        m7 = metrics["M7_token_overhead"]

        m4_str = f"{m4['rate']:.2%}" if m4["rate"] is not None else "N/A"
        m6_str = f"{m6['mean_rtt_s']:.2f}s" if m6["mean_rtt_s"] is not None else "N/A"
        m7_str = f"{m7['avg_total_tokens_per_episode']:.0f}"
        vlm_str = f"{m6['total_vlm_calls']}"

        print(f"{name:<28} {m4_str:>10} {m6_str:>10} {m7_str:>12} {vlm_str:>10}")

    print()

    # Pairwise comparisons
    if pairwise:
        print("-" * 100)
        print(f"Statistical Comparisons (C0 vs Baselines, Bonferroni threshold: p < {BONFERRONI_THRESHOLD:.4f}):")
        print(f"{'Comparison':<22} {'Metric':<20} {'p-value':>10} {'OR':>8} {'Sig?':>6}")
        print("-" * 100)
        for comp_name, comp_data in pairwise.items():
            for metric_name, test_result in comp_data.items():
                p_val = test_result.get("p_value")
                odds = test_result.get("odds_ratio")
                sig = test_result.get("significant_bonferroni")

                p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
                or_str = f"{odds:.2f}" if odds is not None and odds != float("inf") else "inf" if odds == float("inf") else "N/A"
                sig_str = "***" if sig else ""

                print(f"{comp_name:<22} {metric_name:<20} {p_str:>10} {or_str:>8} {sig_str:>6}")
        print()

    # Notes
    print(sep)
    print("Notes:")
    print("  - M1: Task success = all subgoals completed (monitor_complete).")
    print("  - C1 (naive): no subgoals; task success requires manual video review.")
    print("  - C3 (open loop): convergence is self-fulfilling; task success requires manual video review.")
    print("  - M3: C1 and C3 excluded from subgoal success (no reliable automated signal).")
    print("  - M5 (qualitative): Requires manual video annotation, not computed here.")
    print(f"  - Bonferroni correction applied for 6 pairwise comparisons (alpha = {BONFERRONI_THRESHOLD:.4f}).")
    if not HAS_SCIPY:
        print("  - [WARNING] scipy not installed. Statistical tests were skipped.")
    print(sep)


def print_category_table(category_metrics: dict[str, dict[str, dict]]):
    """Print a per-category breakdown showing how each condition performs
    on sequential vs constrained tasks (M1, M2, M3)."""
    sep = "=" * 110
    print()
    print(sep)
    print(f"{'Per-Category Breakdown (Sequential vs Constrained)':^110}")
    print(sep)
    print()

    # Collect all categories across conditions
    all_cats = set()
    for cond_data in category_metrics.values():
        all_cats.update(cond_data.keys())
    all_cats = sorted(all_cats)

    for cat in all_cats:
        print(f"Category: {cat}")
        header = (
            f"  {'Condition':<28} {'N':>4} "
            f"{'M1 (Task)':>12} {'M1 CI':>20} "
            f"{'M2 (Const)':>12} {'M2 CI':>20} "
            f"{'M3 (Subg)':>12}"
        )
        print(header)
        print("  " + "-" * 106)

        for cond in CONDITION_ORDER:
            cond_data = category_metrics.get(cond, {})
            cat_data = cond_data.get(cat)
            if not cat_data:
                continue

            name = CONDITION_NAMES.get(cond, cond)
            n = cat_data["n_episodes"]
            m1 = cat_data["M1_task_success"]
            m2 = cat_data["M2_constraint_adherence"]
            m3 = cat_data["M3_subgoal_success"]

            m1_str = f"{m1['rate']:.2%}" if m1["determined"] > 0 else "N/A"
            m1_ci = (
                f"[{m1['wilson_ci_lower']:.3f}, {m1['wilson_ci_upper']:.3f}]"
                if m1["determined"] > 0 else "N/A"
            )
            m2_str = f"{m2['rate']:.2%}" if m2["determined"] > 0 else "N/A"
            m2_ci = (
                f"[{m2['wilson_ci_lower']:.3f}, {m2['wilson_ci_upper']:.3f}]"
                if m2["determined"] > 0 else "N/A"
            )
            m3_str = f"{m3['rate']:.2%}" if m3["total_subgoals"] > 0 else "N/A"

            print(
                f"  {name:<28} {n:>4} "
                f"{m1_str:>12} {m1_ci:>20} "
                f"{m2_str:>12} {m2_ci:>20} "
                f"{m3_str:>12}"
            )

        print()

    print(sep)
    print()


def print_map_table(map_metrics: dict[str, dict[str, dict]]):
    """Print a per-map breakdown showing generalization across environments."""
    sep = "=" * 110
    print()
    print(sep)
    print(f"{'Per-Map Breakdown (Generalization Across Environments)':^110}")
    print(sep)
    print()

    all_maps = set()
    for cond_data in map_metrics.values():
        all_maps.update(cond_data.keys())
    all_maps = sorted(all_maps)

    for map_name in all_maps:
        print(f"Map: {map_name}")
        header = (
            f"  {'Condition':<28} {'N':>4} "
            f"{'M1 (Task)':>12} {'M1 CI':>20} "
            f"{'M2 (Const)':>12} {'M2 CI':>20} "
            f"{'M3 (Subg)':>12}"
        )
        print(header)
        print("  " + "-" * 106)

        for cond in CONDITION_ORDER:
            cond_data = map_metrics.get(cond, {})
            md = cond_data.get(map_name)
            if not md:
                continue

            name = CONDITION_NAMES.get(cond, cond)
            n = md["n_episodes"]
            m1 = md["M1_task_success"]
            m2 = md["M2_constraint_adherence"]
            m3 = md["M3_subgoal_success"]

            m1_str = f"{m1['rate']:.2%}" if m1["determined"] > 0 else "N/A"
            m1_ci = (
                f"[{m1['wilson_ci_lower']:.3f}, {m1['wilson_ci_upper']:.3f}]"
                if m1["determined"] > 0 else "N/A"
            )
            m2_str = f"{m2['rate']:.2%}" if m2["determined"] > 0 else "N/A"
            m2_ci = (
                f"[{m2['wilson_ci_lower']:.3f}, {m2['wilson_ci_upper']:.3f}]"
                if m2["determined"] > 0 else "N/A"
            )
            m3_str = f"{m3['rate']:.2%}" if m3["total_subgoals"] > 0 else "N/A"

            print(
                f"  {name:<28} {n:>4} "
                f"{m1_str:>12} {m1_ci:>20} "
                f"{m2_str:>12} {m2_ci:>20} "
                f"{m3_str:>12}"
            )

        print()

    print(sep)
    print()


def print_mcnemar_table(mcnemar_results: dict[str, dict]):
    """Print McNemar's test results for paired comparisons."""
    if not mcnemar_results:
        return
    sep = "-" * 100
    print()
    print(sep)
    print(f"McNemar's Test (Paired C0 vs Baselines, Bonferroni threshold: p < {BONFERRONI_THRESHOLD:.4f}):")
    print(f"{'Comparison':<22} {'Metric':<28} {'p-value':>10} {'Discordant':>11} {'Paired':>7} {'Sig?':>6}")
    print(sep)
    for comp_name, comp_data in mcnemar_results.items():
        for metric_name, test_result in comp_data.items():
            p_val = test_result.get("p_value")
            n_disc = test_result.get("n_discordant", "")
            n_paired = test_result.get("n_paired", "")
            sig = test_result.get("significant_bonferroni")
            note = test_result.get("note", "")

            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            sig_str = "***" if sig else ""
            label = metric_name
            if note:
                label = f"{metric_name} ({note})"

            print(f"{comp_name:<22} {label:<28} {p_str:>10} {str(n_disc):>11} {str(n_paired):>7} {sig_str:>6}")
    print()


def write_json_output(
    condition_metrics: dict[str, dict],
    pairwise: dict[str, dict],
    output_path: Path,
    category_metrics: dict[str, dict[str, dict]] | None = None,
    map_metrics: dict[str, dict[str, dict]] | None = None,
    mcnemar_results: dict[str, dict] | None = None,
):
    """Write full metrics to JSON."""
    output = {
        "metadata": {
            "script": "aggregate_results.py",
            "bonferroni_threshold": BONFERRONI_THRESHOLD,
            "scipy_available": HAS_SCIPY,
        },
        "conditions": condition_metrics,
        "pairwise_comparisons": pairwise,
    }
    if category_metrics:
        output["category_breakdown"] = category_metrics
    if map_metrics:
        output["map_breakdown"] = map_metrics
    if mcnemar_results:
        output["mcnemar_paired_tests"] = mcnemar_results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON output written to: {output_path}")


def write_csv_output(
    condition_metrics: dict[str, dict],
    pairwise: dict[str, dict],
    output_path: Path,
):
    """Write summary metrics to CSV for LaTeX/matplotlib import."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "condition", "condition_label", "n_episodes",
        "m1_rate", "m1_ci_lower", "m1_ci_upper", "m1_successes", "m1_determined",
        "m2_rate", "m2_ci_lower", "m2_ci_upper", "m2_adhered", "m2_determined",
        "m3_rate", "m3_ci_lower", "m3_ci_upper", "m3_successful", "m3_total",
        "m4_rate", "m4_total_convergences", "m4_rescued",
        "m6_mean_rtt_s", "m6_median_rtt_s", "m6_p95_rtt_s", "m6_total_calls",
        "m7_avg_input_tokens", "m7_avg_output_tokens", "m7_avg_total_tokens",
        "m8_avg_steps", "m8_avg_wall_clock_s",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cond in CONDITION_ORDER:
            metrics = condition_metrics.get(cond)
            if not metrics:
                continue

            m1 = metrics["M1_task_success"]
            m2 = metrics["M2_constraint_adherence"]
            m3 = metrics["M3_subgoal_success"]
            m4 = metrics["M4_correction_rate"]
            m6 = metrics["M6_latency"]
            m7 = metrics["M7_token_overhead"]
            m8 = metrics["M8_episode_length"]

            row = {
                "condition": cond,
                "condition_label": CONDITION_NAMES.get(cond, cond),
                "n_episodes": metrics["n_episodes"],
                "m1_rate": m1["rate"],
                "m1_ci_lower": m1["wilson_ci_lower"],
                "m1_ci_upper": m1["wilson_ci_upper"],
                "m1_successes": m1["successes"],
                "m1_determined": m1["determined"],
                "m2_rate": m2["rate"],
                "m2_ci_lower": m2["wilson_ci_lower"],
                "m2_ci_upper": m2["wilson_ci_upper"],
                "m2_adhered": m2["adhered"],
                "m2_determined": m2["determined"],
                "m3_rate": m3["rate"],
                "m3_ci_lower": m3["wilson_ci_lower"],
                "m3_ci_upper": m3["wilson_ci_upper"],
                "m3_successful": m3["successful_subgoals"],
                "m3_total": m3["total_subgoals"],
                "m4_rate": m4["rate"] if m4["rate"] is not None else "",
                "m4_total_convergences": m4["total_premature_convergences"],
                "m4_rescued": m4["rescued_convergences"],
                "m6_mean_rtt_s": m6["mean_rtt_s"] if m6["mean_rtt_s"] is not None else "",
                "m6_median_rtt_s": m6["median_rtt_s"] if m6["median_rtt_s"] is not None else "",
                "m6_p95_rtt_s": m6["p95_rtt_s"] if m6["p95_rtt_s"] is not None else "",
                "m6_total_calls": m6["total_vlm_calls"],
                "m7_avg_input_tokens": m7["avg_input_tokens_per_episode"],
                "m7_avg_output_tokens": m7["avg_output_tokens_per_episode"],
                "m7_avg_total_tokens": m7["avg_total_tokens_per_episode"],
                "m8_avg_steps": m8["avg_steps"],
                "m8_avg_wall_clock_s": m8["avg_wall_clock_s"],
            }
            writer.writerow(row)

    # Also write pairwise comparisons CSV
    pairwise_path = output_path.parent / "pairwise_comparisons.csv"
    pw_fields = ["comparison", "metric", "p_value", "odds_ratio", "significant_bonferroni"]
    with open(pairwise_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pw_fields)
        writer.writeheader()
        for comp_name, comp_data in pairwise.items():
            for metric_name, test_result in comp_data.items():
                writer.writerow({
                    "comparison": comp_name,
                    "metric": metric_name,
                    "p_value": test_result.get("p_value", ""),
                    "odds_ratio": test_result.get("odds_ratio", ""),
                    "significant_bonferroni": test_result.get("significant_bonferroni", ""),
                })

    print(f"CSV output written to: {output_path}")
    print(f"Pairwise CSV written to: {pairwise_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate RVLN experimental results and compute metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Path to results directory (default: results/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output files (default: same as results_dir)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific conditions (e.g. condition0 condition1)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    print(f"Scanning results in: {os.path.abspath(results_dir)}")
    print()

    # Load all runs
    runs_by_condition = find_run_infos(results_dir, args.conditions)

    if not runs_by_condition:
        print("No results found. Check --results_dir path.", file=sys.stderr)
        sys.exit(1)

    # Report what was found
    print("Episodes found per condition:")
    for cond in CONDITION_ORDER:
        n = len(runs_by_condition.get(cond, []))
        if n > 0:
            label = CONDITION_NAMES.get(cond, cond)
            print(f"  {label}: {n} episodes")
    print()

    # Compute per-condition metrics
    condition_metrics = {}
    for cond in CONDITION_ORDER:
        runs = runs_by_condition.get(cond, [])
        if runs:
            condition_metrics[cond] = aggregate_condition(cond, runs)

    # Compute pairwise statistical tests
    pairwise = compute_pairwise_tests(condition_metrics)

    # Compute McNemar's test (paired by task_id)
    mcnemar_results = compute_paired_mcnemar_tests(runs_by_condition)

    # Compute per-category breakdown (sequential vs constrained)
    category_metrics = aggregate_by_category(runs_by_condition)

    # Compute per-map breakdown (generalization across environments)
    map_metrics = aggregate_by_map(runs_by_condition)

    # Output
    print_summary_table(condition_metrics, pairwise)
    print_mcnemar_table(mcnemar_results)
    print_category_table(category_metrics)
    print_map_table(map_metrics)

    # Write files
    output_path = Path(output_dir)
    write_json_output(
        condition_metrics, pairwise, output_path / "experiment_summary.json",
        category_metrics=category_metrics,
        map_metrics=map_metrics,
        mcnemar_results=mcnemar_results,
    )
    write_csv_output(condition_metrics, pairwise, output_path / "experiment_summary.csv")


if __name__ == "__main__":
    main()
