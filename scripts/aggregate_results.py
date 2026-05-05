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

Known limitations:
  - Category imbalance: Greek Island is ~80% constrained tasks, Suburb
    Neighborhood is ~80% sequential. Per-category breakdowns are confounded
    by map effects.
  - Statistical power: 45 episodes per condition (15 tasks x 3 starting
    positions) is borderline for detecting moderate effect sizes. Starting
    position variants may not be fully independent, reducing effective N.
  - Stochastic factors: simulator determinism is imperfect (propeller visual
    effects, rendering variations) and cannot be fully controlled.
  - Manual annotation: C1 and C3 require manual video review for task
    success (M1) and constraint adherence (M2). Use --strict to enforce.
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

# Bonferroni-corrected significance threshold (0.05 / 6 comparisons).
# Applied to the M1 headline test family only; M2 / M3 are reported as
# secondary analyses without a family-wise correction claim.
BONFERRONI_THRESHOLD = 0.05 / 6  # ~0.0083

# M4 (supervisor correction rate) is only meaningful for conditions whose
# pipeline includes the GoalAdherenceMonitor's convergence flow. C1 has no
# subgoal decomposition; C3 is open-loop with no monitor at all.
M4_APPLICABLE_CONDITIONS = {
    "condition0", "condition2", "condition4", "condition5", "condition6",
}

API_PRICING = {
    "gpt-5.4": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gemini-2.0-flash": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
}
DEFAULT_PRICING = {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000}


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
    family: str = "primary",
) -> dict[str, Any]:
    """
    Run Fisher's exact test comparing C0 to another condition.

    ``family`` controls which significance flag is reported:
      * ``"primary"`` (default): used for the headline M1 family. Reports
        ``significant_bonferroni`` against ``BONFERRONI_THRESHOLD = 0.05/6``.
      * ``"exploratory"``: used for M2 / M3 secondary analyses. Reports
        the raw p-value and a ``significant_uncorrected`` flag at
        alpha=0.05; no family-wise correction claim is made because these
        comparisons are not the headline test.

    Returns dict with p_value, odds_ratio, and the appropriate significance
    flag.
    """
    if not HAS_SCIPY:
        return {
            "p_value": None,
            "odds_ratio": None,
            "significant_bonferroni": None,
            "significant_uncorrected": None,
            "family": family,
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
        "family": family,
        "significant_bonferroni": (
            p_value < BONFERRONI_THRESHOLD if family == "primary" else None
        ),
        "significant_uncorrected": (
            p_value < 0.05 if family == "exploratory" else None
        ),
    }


def mcnemars_test(
    c0_outcomes: list[bool | None],
    cx_outcomes: list[bool | None],
) -> dict[str, Any]:
    """
    McNemar's test for paired binary outcomes (matched starting positions).

    Each list entry corresponds to the same episode (same task + same starting
    position) run under two conditions.  Entries where either outcome is None
    (manual review still pending) are excluded from the test and counted in
    ``n_skipped``. When ALL pairs are skipped the result reports
    ``p_value=None`` with ``n_paired=0``, distinguishing "untestable" from
    "tested and not significant" in the output.

    Returns dict with statistic, p_value, n_discordant, n_skipped, and
    significance flag.
    """
    n_total = len(c0_outcomes)
    n_skipped = sum(
        1 for o0, ox in zip(c0_outcomes, cx_outcomes)
        if o0 is None or ox is None
    )
    if n_skipped == n_total and n_total > 0:
        return {
            "statistic": None,
            "p_value": None,
            "n_paired": 0,
            "n_skipped": n_skipped,
            "n_discordant": 0,
            "significant_bonferroni": None,
            "note": "all pairs skipped (missing annotation); test untestable",
        }
    if not HAS_SCIPY:
        return {
            "statistic": None,
            "p_value": None,
            "n_paired": n_total - n_skipped,
            "n_skipped": n_skipped,
            "n_discordant": None,
            "significant_bonferroni": None,
            "note": "scipy not installed",
        }

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
            "n_skipped": n_skipped,
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
            "n_skipped": n_skipped,
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
        "n_skipped": n_skipped,
        "n_discordant": n_discordant,
        "significant_bonferroni": p_value < BONFERRONI_THRESHOLD,
    }


def wilcoxon_paired(
    c0_rates: dict[str, float],
    cx_rates: dict[str, float],
) -> dict[str, Any]:
    """
    Paired two-sided Wilcoxon signed-rank test on per-episode rates.

    Used for M3 (per-episode subgoal success rate) where the underlying
    outcomes (subgoal completions) are clustered within episodes and an
    unpaired Fisher's test would inflate Type I error. Pairs are matched by
    the run pair key ``(task_id, tuple(initial_pos))``.

    Returns dict with statistic, p_value, n_paired, n_nonzero, n_skipped,
    and the median paired difference.
    """
    pair_keys = sorted(set(c0_rates) & set(cx_rates))
    n_skipped = (len(c0_rates) + len(cx_rates)) // 2 - len(pair_keys)
    n_paired = len(pair_keys)
    if n_paired == 0:
        return {
            "statistic": None,
            "p_value": None,
            "n_paired": 0,
            "n_skipped": n_skipped,
            "n_nonzero": 0,
            "median_diff": None,
            "note": "no overlapping pairs",
        }
    if not HAS_SCIPY:
        return {
            "statistic": None,
            "p_value": None,
            "n_paired": n_paired,
            "n_skipped": n_skipped,
            "n_nonzero": None,
            "median_diff": None,
            "note": "scipy not installed",
        }
    diffs = [c0_rates[k] - cx_rates[k] for k in pair_keys]
    nonzero = [d for d in diffs if d != 0]
    median_diff = sorted(diffs)[len(diffs) // 2]
    if not nonzero:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_paired": n_paired,
            "n_skipped": n_skipped,
            "n_nonzero": 0,
            "median_diff": round(median_diff, 4),
            "note": "all paired differences are zero",
        }
    try:
        result = scipy_stats.wilcoxon(
            [c0_rates[k] for k in pair_keys],
            [cx_rates[k] for k in pair_keys],
            zero_method="wilcox", alternative="two-sided",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError as e:
        return {
            "statistic": None,
            "p_value": None,
            "n_paired": n_paired,
            "n_skipped": n_skipped,
            "n_nonzero": len(nonzero),
            "median_diff": round(median_diff, 4),
            "note": f"wilcoxon failed: {e}",
        }
    return {
        "statistic": round(statistic, 4),
        "p_value": p_value,
        "n_paired": n_paired,
        "n_skipped": n_skipped,
        "n_nonzero": len(nonzero),
        "median_diff": round(median_diff, 4),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

EXPECTED_RUNS_PER_CONDITION = 45  # 15 tasks x 3 starting-position variants


def find_run_infos(
    results_dir: str,
    conditions: list[str] | None = None,
    expected_n: int | None = EXPECTED_RUNS_PER_CONDITION,
    strict: bool = False,
) -> dict[str, list[dict]]:
    """
    Scan results directory and load all run_info.json files, grouped by condition.

    Handles two directory structures:
      - results/conditionN/<map_dir>/<run_name>/run_info.json  (multi-map)
      - results/conditionN/<run_name>/run_info.json            (legacy/flat)

    Filters: aborted (``aborted=True``) and incomplete (``completed=False``)
    runs are excluded so half-finished or crashed episodes don't pollute the
    metrics. When ``expected_n`` is set, a count mismatch triggers a warning
    (or a hard error under ``strict=True``) so missing cells are visible
    rather than silently shrinking the denominator.
    """
    results_path = Path(results_dir)
    runs_by_condition: dict[str, list[dict]] = defaultdict(list)
    skip_counts: dict[str, dict[str, int]] = defaultdict(lambda: {
        "aborted": 0, "incomplete": 0, "load_error": 0,
    })

    for cond_dir in sorted(results_path.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond_name = cond_dir.name
        if not cond_name.startswith("condition"):
            continue
        if conditions and cond_name not in conditions:
            continue

        for run_info_path in cond_dir.rglob("run_info.json"):
            try:
                with open(run_info_path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [WARN] Could not load {run_info_path}: {e}", file=sys.stderr)
                skip_counts[cond_name]["load_error"] += 1
                continue
            if data.get("aborted", False):
                skip_counts[cond_name]["aborted"] += 1
                continue
            if not data.get("completed", False):
                skip_counts[cond_name]["incomplete"] += 1
                continue
            data["_source_path"] = str(run_info_path)
            constraint_path = run_info_path.parent / "constraint_analysis.json"
            if constraint_path.exists():
                try:
                    with open(constraint_path, "r") as f:
                        data["_constraint_analysis"] = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    print(f"  [WARN] Could not load {constraint_path}: {e}",
                          file=sys.stderr)
            runs_by_condition[cond_name].append(data)

    # Visibility on filtered / missing cells.
    for cond_name in sorted({*runs_by_condition.keys(), *skip_counts.keys()}):
        n = len(runs_by_condition.get(cond_name, []))
        sk = skip_counts[cond_name]
        skipped_total = sk["aborted"] + sk["incomplete"] + sk["load_error"]
        if skipped_total:
            print(
                f"  [INFO] {cond_name}: kept {n}, skipped "
                f"{sk['aborted']} aborted, {sk['incomplete']} incomplete, "
                f"{sk['load_error']} load_error",
                file=sys.stderr,
            )
        if expected_n is not None and n != expected_n:
            msg = (
                f"{cond_name}: expected {expected_n} completed runs, "
                f"found {n} (skipped {skipped_total})"
            )
            if strict:
                raise RuntimeError(msg)
            print(f"  [WARN] {msg}", file=sys.stderr)

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


def is_constraint_adhered(run: dict, require_manual: bool = False) -> bool | None:
    """
    Check constraint adherence (M2) for an episode.

    Per the experimental design (Section 6f), the official M2 source is a
    *manual* video review for ALL 7 conditions, stored as
    ``constraint_analysis.json`` next to the run's ``run_info.json``. The
    runtime monitor's ``any_constraint_violated`` flag is logged for
    diagnostic comparison but is not the headline source.

    Behaviour:
      * ``require_manual=True`` (used under --strict): only honour
        ``_constraint_analysis``. Return ``None`` whenever the manual
        annotation is missing, so missing-annotation cells are visible in
        the report rather than silently filled in by the runtime flag.
      * ``require_manual=False`` (default): fall back to the runtime
        ``any_constraint_violated`` flag and per-subgoal counters. Useful
        for early development before annotations exist; do NOT publish
        these numbers without flagging the fallback.
    """
    ca = run.get("_constraint_analysis")
    if ca:
        if "any_constraint_violated" in ca:
            return not ca["any_constraint_violated"]
        if "constraint_adherence" in ca:
            return bool(ca["constraint_adherence"])

    if require_manual:
        return None

    acv = run.get("any_constraint_violated")
    if acv is not None:
        return not acv

    # If any_constraint_violated is explicitly None, constraints were not
    # checked at runtime (C1, C2, C3). Needs post-hoc analysis.
    if "any_constraint_violated" in run and acv is None:
        return None

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

    A *rescue* requires (a) at least one convergence event during the
    subgoal and (b) the subgoal to ultimately finish as ``monitor_complete``.
    Other terminal stop reasons (``max_steps``, ``ask_help``, ``abort``,
    ``skipped``, ``convergence_error``) indicate the supervisor failed to
    rescue the subgoal and contribute zero rescued convergences. The final
    convergence in a successful chain is the one that yielded completion;
    only the earlier ``convergence_count - 1`` events count as rescues.

    Returns dict with total_convergences, rescued_convergences, corrections_used.
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
        records = sg.get("vlm_call_records", [])
        convergence_count = sum(
            1 for r in records if r.get("label", "").startswith("convergence")
        )
        total_convergences += convergence_count
        stop = sg.get("stop_reason", "")
        if stop == "monitor_complete" and convergence_count > 0:
            # The final convergence ended in completion; earlier ones were
            # genuine rescues (corrective command issued, drone resumed
            # progress, eventually completed).
            rescued_convergences += max(0, convergence_count - 1)
        # Any other stop reason: zero rescues. Even if there were
        # convergence events, they did not lead to completion, so they are
        # failed-rescue attempts, not rescues.

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


def compute_path_efficiency(run: dict) -> dict | None:
    """
    Compute path efficiency as displacement / total_distance.

    Tries trajectory_log first (inlined or on-disk), then falls back to
    subgoal_summaries origins.  Returns None when position data is
    unavailable or total_distance is zero.
    """
    positions: list[tuple[float, float, float]] = []

    # Try inlined trajectory log
    traj_log = run.get("_trajectory_log")
    if traj_log and isinstance(traj_log, list):
        for entry in traj_log:
            pos = entry.get("position")
            if pos and len(pos) >= 2:
                positions.append(tuple(float(v) for v in pos[:3]) if len(pos) >= 3 else (float(pos[0]), float(pos[1]), 0.0))

    # Try loading trajectory log from disk
    if not positions:
        source_path = run.get("_source_path", "")
        if source_path:
            traj_path = Path(source_path).parent / "trajectory_log.json"
            if traj_path.exists():
                try:
                    with open(traj_path, "r") as f:
                        traj_log = json.load(f)
                    if isinstance(traj_log, list):
                        for entry in traj_log:
                            pos = entry.get("position")
                            if pos and len(pos) >= 2:
                                positions.append(
                                    tuple(float(v) for v in pos[:3]) if len(pos) >= 3
                                    else (float(pos[0]), float(pos[1]), 0.0)
                                )
                except (json.JSONDecodeError, OSError):
                    pass

    # Fall back to subgoal_summaries origins
    if not positions:
        summaries = run.get("subgoal_summaries", [])
        for sg in summaries:
            origin = sg.get("next_origin")
            if origin and len(origin) >= 2:
                positions.append(
                    tuple(float(v) for v in origin[:3]) if len(origin) >= 3
                    else (float(origin[0]), float(origin[1]), 0.0)
                )

    if len(positions) < 2:
        return None

    total_distance = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dz = positions[i][2] - positions[i - 1][2]
        total_distance += math.sqrt(dx * dx + dy * dy + dz * dz)

    if total_distance == 0:
        return None

    first, last = positions[0], positions[-1]
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    dz = last[2] - first[2]
    displacement = math.sqrt(dx * dx + dy * dy + dz * dz)

    efficiency_ratio = min(displacement / total_distance, 1.0)

    return {
        "total_distance": round(total_distance, 4),
        "displacement": round(displacement, 4),
        "efficiency_ratio": round(efficiency_ratio, 4),
    }


def compute_api_cost(run: dict) -> dict:
    """
    Estimate API cost for an episode based on token counts and model pricing.
    """
    model = run.get("monitor_model") or run.get("llm_model") or ""
    pricing = API_PRICING.get(model, DEFAULT_PRICING)
    pricing_source = "known" if model in API_PRICING else "default"

    input_tokens = run.get("total_input_tokens", 0)
    output_tokens = run.get("total_output_tokens", 0)

    input_cost = input_tokens * pricing["input"]
    output_cost = output_tokens * pricing["output"]

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
        "model": model,
        "pricing_source": pricing_source,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_condition(
    condition: str, runs: list[dict], require_manual_m2: bool = False,
) -> dict[str, Any]:
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
        result = is_constraint_adhered(run, require_manual=require_manual_m2)
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

    # M4: Supervisor Correction Rate (gated to applicable conditions only).
    # M4 is only meaningful for conditions that include the supervisor /
    # convergence-correction loop. C1 has no decomposition, C3 is open-loop;
    # both are excluded so they don't contribute spurious zero-denominators
    # or zero-numerators to the headline rate.
    total_convergences = 0
    total_rescued = 0
    total_corrections = 0
    if condition in M4_APPLICABLE_CONDITIONS:
        for run in runs:
            cr = compute_correction_rate(run)
            if cr:
                total_convergences += cr["total_convergences"]
                total_rescued += cr["rescued_convergences"]
                total_corrections += cr["total_corrections"]

    if condition not in M4_APPLICABLE_CONDITIONS:
        m4_rate = None
    elif total_convergences > 0:
        m4_rate = total_rescued / total_convergences
    else:
        m4_rate = None

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

    # Path efficiency (per-episode, then averaged)
    efficiency_values = []
    for run in runs:
        pe = compute_path_efficiency(run)
        if pe is not None:
            efficiency_values.append(pe["efficiency_ratio"])

    avg_efficiency = (
        round(sum(efficiency_values) / len(efficiency_values), 4)
        if efficiency_values else None
    )

    # M9: API cost
    total_cost = 0.0
    total_input_cost = 0.0
    total_output_cost = 0.0
    cost_model = None
    cost_pricing_source = None
    for run in runs:
        ac = compute_api_cost(run)
        total_cost += ac["total_cost"]
        total_input_cost += ac["input_cost"]
        total_output_cost += ac["output_cost"]
        if cost_model is None:
            cost_model = ac["model"]
            cost_pricing_source = ac["pricing_source"]

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
            # Per-episode rates keyed by (task_id, initial_pos) for paired
            # tests downstream. Episodes without a determinable rate are
            # omitted; the pair key matches _run_pair_key().
            "per_episode_rates": {
                _run_pair_key(run): rate
                for run, rate in zip(runs, [
                    compute_subgoal_success_rate(r) for r in runs
                ])
                if rate is not None
            },
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
        "path_efficiency": {
            "avg_efficiency_ratio": avg_efficiency,
            "n_episodes_with_data": len(efficiency_values),
        },
        "M9_api_cost": {
            "total_cost": round(total_cost, 4),
            "total_input_cost": round(total_input_cost, 4),
            "total_output_cost": round(total_output_cost, 4),
            "avg_cost_per_episode": round(total_cost / n, 4),
            "model": cost_model,
            "pricing_source": cost_pricing_source,
        },
    }


def aggregate_by_category(
    runs_by_condition: dict[str, list[dict]],
    require_manual_m2: bool = False,
) -> dict[str, dict[str, dict]]:
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
                res = is_constraint_adhered(run, require_manual=require_manual_m2)
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


_ENV_ID_TO_TASK_DIR = {
    "UnrealTrack-DowntownWest-ContinuousColor-v0": "downtown_west",
    "UnrealTrack-Greek_Island-ContinuousColor-v0": "greek_island",
    "UnrealTrack-SuburbNeighborhood_Day-ContinuousColor-v0": "suburb_neighborhood_day",
}


def _normalize_map_name(run: dict) -> str:
    """Return a canonical task_dir_name for a run.

    env_id strings (e.g. ``UnrealTrack-Greek_Island-ContinuousColor-v0``) and
    on-disk path components (``greek_island``) are mapped to the same
    canonical label so per-map breakdowns aren't split across naming styles.
    """
    env_id = run.get("env_id", "") or ""
    if env_id in _ENV_ID_TO_TASK_DIR:
        return _ENV_ID_TO_TASK_DIR[env_id]
    source = run.get("_source_path", "")
    parts = Path(source).parts
    for i, p in enumerate(parts):
        if p.startswith("condition"):
            if i + 1 < len(parts) and not parts[i + 1].startswith("c"):
                return parts[i + 1]
            break
    if env_id:
        return env_id.lower()
    return "unknown"


def aggregate_by_map(
    runs_by_condition: dict[str, list[dict]],
    require_manual_m2: bool = False,
) -> dict[str, dict[str, dict]]:
    """
    Group episodes by map and compute M1/M2/M3 for each map within each
    condition.

    Map labels are normalised to the task_dir_name (e.g. ``greek_island``)
    via :func:`_normalize_map_name` so env_id strings and on-disk path
    components don't produce duplicate buckets.

    Returns: {condition: {map_name: {M1, M2, M3 metrics}}}.
    """
    result: dict[str, dict[str, dict]] = {}

    for cond in CONDITION_ORDER:
        runs = runs_by_condition.get(cond, [])
        if not runs:
            continue

        by_map: dict[str, list[dict]] = defaultdict(list)
        for run in runs:
            map_name = _normalize_map_name(run)
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
                res = is_constraint_adhered(run, require_manual=require_manual_m2)
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


def aggregate_by_map_and_category(
    runs_by_condition: dict[str, list[dict]],
    require_manual_m2: bool = False,
) -> dict[str, dict[str, dict[str, dict]]]:
    """
    Cross-tabulate per-map AND per-category. Returns
    ``{condition: {map: {category: {n, M1, M2, M3}}}}``.

    Motivation: in the current task design, Greek Island is ~80% constrained
    and Suburb Neighborhood is ~80% sequential, so the marginal per-map and
    per-category breakdowns are confounded. Reporting the joint cell is the
    cleanest way to disentangle "LTL helps on constrained tasks" from "LTL
    helps in Greek Island". The paper should lead with this cross-tab.
    """
    result: dict[str, dict[str, dict[str, dict]]] = {}

    for cond in CONDITION_ORDER:
        runs = runs_by_condition.get(cond, [])
        if not runs:
            continue

        cells: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list),
        )
        for run in runs:
            map_name = _normalize_map_name(run)
            category = run.get("task", {}).get("category", "uncategorized")
            cells[map_name][category].append(run)

        cond_out: dict[str, dict[str, dict]] = {}
        for map_name in sorted(cells):
            map_out: dict[str, dict] = {}
            for category in sorted(cells[map_name]):
                cell_runs = cells[map_name][category]

                m1_succ = m1_det = 0
                for run in cell_runs:
                    res = is_task_success(run)
                    if res is not None:
                        m1_det += 1
                        if res:
                            m1_succ += 1
                m1_rate, m1_lo, m1_hi = wilson_ci(m1_succ, m1_det)

                m2_succ = m2_det = 0
                for run in cell_runs:
                    res = is_constraint_adhered(run, require_manual=require_manual_m2)
                    if res is not None:
                        m2_det += 1
                        if res:
                            m2_succ += 1
                m2_rate, m2_lo, m2_hi = wilson_ci(m2_succ, m2_det)

                m3_succ = m3_total = 0
                for run in cell_runs:
                    c = run.get("condition", "")
                    if "condition3" in c or "open_loop" in c:
                        continue
                    for sg in run.get("subgoal_summaries", []):
                        m3_total += 1
                        if sg.get("stop_reason", "") == "monitor_complete":
                            m3_succ += 1
                m3_rate, m3_lo, m3_hi = wilson_ci(m3_succ, m3_total)

                map_out[category] = {
                    "n_episodes": len(cell_runs),
                    "M1_task_success": {
                        "successes": m1_succ,
                        "determined": m1_det,
                        "rate": round(m1_rate, 4),
                        "wilson_ci_lower": round(m1_lo, 4),
                        "wilson_ci_upper": round(m1_hi, 4),
                    },
                    "M2_constraint_adherence": {
                        "adhered": m2_succ,
                        "determined": m2_det,
                        "rate": round(m2_rate, 4),
                        "wilson_ci_lower": round(m2_lo, 4),
                        "wilson_ci_upper": round(m2_hi, 4),
                    },
                    "M3_subgoal_success": {
                        "successful_subgoals": m3_succ,
                        "total_subgoals": m3_total,
                        "rate": round(m3_rate, 4),
                        "wilson_ci_lower": round(m3_lo, 4),
                        "wilson_ci_upper": round(m3_hi, 4),
                    },
                }
            cond_out[map_name] = map_out
        result[cond] = cond_out

    return result


def aggregate_by_subgoal_position(
    runs_by_condition: dict[str, list[dict]],
) -> dict[str, dict[int, dict]]:
    """
    For each condition, group subgoals by ordinal position across episodes
    and compute Wilson CI success rates per position.

    Skips C1 (no subgoals) and C3 (open-loop convergence is unreliable).

    Returns: {condition: {position_idx: {successes, total, rate,
              wilson_ci_lower, wilson_ci_upper}}}
    """
    skip_conditions = {"condition1", "condition3"}
    result: dict[str, dict[int, dict]] = {}

    for cond in CONDITION_ORDER:
        if cond in skip_conditions:
            continue
        runs = runs_by_condition.get(cond, [])
        if not runs:
            continue

        position_data: dict[int, dict] = defaultdict(lambda: {"successes": 0, "total": 0})

        for run in runs:
            summaries = run.get("subgoal_summaries", [])
            for idx, sg in enumerate(summaries):
                position_data[idx]["total"] += 1
                if sg.get("stop_reason", "") == "monitor_complete":
                    position_data[idx]["successes"] += 1

        cond_positions = {}
        for pos_idx in sorted(position_data.keys()):
            pd = position_data[pos_idx]
            rate, ci_lo, ci_hi = wilson_ci(pd["successes"], pd["total"])
            cond_positions[pos_idx] = {
                "successes": pd["successes"],
                "total": pd["total"],
                "rate": round(rate, 4),
                "wilson_ci_lower": round(ci_lo, 4),
                "wilson_ci_upper": round(ci_hi, 4),
            }

        if cond_positions:
            result[cond] = cond_positions

    return result


def _run_pair_key(run: dict) -> str:
    """Unique JSON-serialisable key for pairing runs across conditions.

    Format: ``"<task_id>|<x>,<y>,<z>,<yaw>"``. Same task_id + same initial
    position implies the same pair across conditions (starting positions are
    fixed across conditions per the experimental design).
    """
    task = run.get("task", {})
    tid = str(task.get("task_id", ""))
    pos = task.get("initial_pos", []) or []
    pos_str = ",".join(f"{float(p):.4f}" for p in pos)
    return f"{tid}|{pos_str}"


def compute_paired_mcnemar_tests(
    runs_by_condition: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    Compute McNemar's test for C0 vs each baseline on M1, using paired
    episodes matched by (task_id, initial_pos).

    Starting positions are fixed across conditions, so episodes with the
    same task_id and starting position form a natural pair.
    """
    comparisons = {}
    c0_runs = runs_by_condition.get("condition0", [])
    if not c0_runs:
        return comparisons

    c0_by_key = {}
    for run in c0_runs:
        key = _run_pair_key(run)
        if key[0]:
            c0_by_key[key] = run

    for cond in CONDITION_ORDER[1:]:
        cx_runs = runs_by_condition.get(cond, [])
        if not cx_runs:
            continue

        cx_by_key = {}
        for run in cx_runs:
            key = _run_pair_key(run)
            if key[0]:
                cx_by_key[key] = run

        shared_keys = sorted(set(c0_by_key.keys()) & set(cx_by_key.keys()))
        if not shared_keys:
            continue

        c0_outcomes = [is_task_success(c0_by_key[k]) for k in shared_keys]
        cx_outcomes = [is_task_success(cx_by_key[k]) for k in shared_keys]

        comparisons[f"C0_vs_{cond}"] = {
            "M1_task_success_mcnemar": mcnemars_test(c0_outcomes, cx_outcomes),
        }

    return comparisons


def compute_pairwise_tests(
    condition_metrics: dict[str, dict],
) -> dict[str, dict]:
    """
    Compute pairwise tests for C0 vs each other condition.

    Significance reporting:
      * M1 (task success) is the headline test family. Bonferroni-corrected
        at 0.05/6 (six C0-vs-baseline comparisons).
      * M2 (constraint adherence) and M3 (per-episode subgoal success rate)
        are secondary / exploratory. Reported with raw p-values at alpha=0.05;
        no family-wise correction claim. M3 uses a paired Wilcoxon test
        (clustering-aware) instead of an unpaired Fisher test.
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

        # M1: primary headline test (Bonferroni-corrected).
        c0_m1 = c0["M1_task_success"]
        cx_m1 = cx["M1_task_success"]
        if c0_m1["determined"] > 0 and cx_m1["determined"] > 0:
            cond_comparisons["M1_task_success"] = fishers_exact_test(
                c0_m1["successes"], c0_m1["determined"],
                cx_m1["successes"], cx_m1["determined"],
                family="primary",
            )

        # M2: exploratory (no family-wise correction).
        c0_m2 = c0["M2_constraint_adherence"]
        cx_m2 = cx["M2_constraint_adherence"]
        if c0_m2["determined"] > 0 and cx_m2["determined"] > 0:
            cond_comparisons["M2_constraint_adherence"] = fishers_exact_test(
                c0_m2["adhered"], c0_m2["determined"],
                cx_m2["adhered"], cx_m2["determined"],
                family="exploratory",
            )

        # M3: paired Wilcoxon signed-rank test on per-episode subgoal-success
        # rates. Fisher's exact test would treat each subgoal as i.i.d.
        # Bernoulli, but subgoals are clustered within episodes (failures
        # cascade once one subgoal fails) and the per-condition denominator
        # depends on early-stopping behaviour, so the independence assumption
        # is violated. Wilcoxon pairs each episode by (task_id, initial_pos)
        # so the within-episode clustering is absorbed into the pair.
        c0_m3 = c0["M3_subgoal_success"]
        cx_m3 = cx["M3_subgoal_success"]
        if c0_m3["total_subgoals"] > 0 and cx_m3["total_subgoals"] > 0:
            cond_comparisons["M3_subgoal_success"] = wilcoxon_paired(
                c0_m3.get("per_episode_rates", {}),
                cx_m3.get("per_episode_rates", {}),
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
    print(
        f"{'Condition':<28} {'M4 (Corr%)':>10} {'M6 (RTT)':>10} "
        f"{'M7 (Tok/ep)':>12} {'VLM Calls':>10} {'M9 ($/ep)':>10}"
    )
    print("-" * 100)
    for cond in CONDITION_ORDER:
        metrics = condition_metrics.get(cond)
        if not metrics:
            continue
        name = CONDITION_NAMES.get(cond, cond)
        m4 = metrics["M4_correction_rate"]
        m6 = metrics["M6_latency"]
        m7 = metrics["M7_token_overhead"]
        m9 = metrics["M9_api_cost"]

        m4_str = f"{m4['rate']:.2%}" if m4["rate"] is not None else "N/A"
        m6_str = f"{m6['mean_rtt_s']:.2f}s" if m6["mean_rtt_s"] is not None else "N/A"
        m7_str = f"{m7['avg_total_tokens_per_episode']:.0f}"
        vlm_str = f"{m6['total_vlm_calls']}"
        m9_str = f"${m9['avg_cost_per_episode']:.4f}"

        print(
            f"{name:<28} {m4_str:>10} {m6_str:>10} "
            f"{m7_str:>12} {vlm_str:>10} {m9_str:>10}"
        )

    print()

    # Pairwise comparisons
    if pairwise:
        print("-" * 100)
        print(
            f"Statistical Comparisons (C0 vs Baselines). "
            f"M1 = primary headline test, Bonferroni-corrected at "
            f"p < {BONFERRONI_THRESHOLD:.4f} (0.05/6). "
            f"M2 / M3 = exploratory secondary analyses, raw p at alpha=0.05."
        )
        print(f"{'Comparison':<22} {'Metric':<22} {'Family':<12} {'p-value':>10} {'Effect':>10} {'Sig?':>6}")
        print("-" * 100)
        for comp_name, comp_data in pairwise.items():
            for metric_name, test_result in comp_data.items():
                p_val = test_result.get("p_value")
                odds = test_result.get("odds_ratio")
                family = test_result.get("family", "primary")
                if family == "primary":
                    sig = test_result.get("significant_bonferroni")
                else:
                    sig = test_result.get("significant_uncorrected")
                # M3 uses Wilcoxon, which reports median_diff instead of OR.
                effect = test_result.get(
                    "median_diff", odds if odds is not None else None,
                )
                p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
                if effect is None:
                    effect_str = "N/A"
                elif effect == float("inf"):
                    effect_str = "inf"
                else:
                    effect_str = f"{effect:.3f}"
                sig_str = "***" if sig else ""
                print(
                    f"{comp_name:<22} {metric_name:<22} {family:<12} "
                    f"{p_str:>10} {effect_str:>10} {sig_str:>6}"
                )
        print()

    # Notes
    print(sep)
    print("Notes:")
    print("  - M1: Task success = all subgoals completed (monitor_complete).")
    print("  - C1 (naive): no subgoals; task success requires manual video review.")
    print("  - C3 (open loop): convergence is self-fulfilling; task success requires manual video review.")
    print("  - M3: C1 and C3 excluded from subgoal success (no reliable automated signal).")
    print("  - M3 statistical test: paired Wilcoxon signed-rank on per-episode rates.")
    print("  - M5 (qualitative): Requires manual video annotation, not computed here.")
    print(f"  - Bonferroni correction applied to M1 family only (6 comparisons, alpha = {BONFERRONI_THRESHOLD:.4f}).")
    print("  - M2 / M3 reported with raw p-values; no family-wise correction claim.")
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


def print_map_x_category_table(
    map_x_category: dict[str, dict[str, dict[str, dict]]],
):
    """Per-map x per-category cross-tab table (the headline figure)."""
    sep = "=" * 120
    print()
    print(sep)
    print(f"{'Per-Map x Per-Category Breakdown (recommended headline table)':^120}")
    print(sep)
    print()
    print("  Reading guide: cells in the same map should be compared across")
    print("  categories to gauge LTL's contribution holding the environment fixed.")
    print()

    all_maps: set[str] = set()
    all_categories: set[str] = set()
    for cond_data in map_x_category.values():
        for map_name, by_cat in cond_data.items():
            all_maps.add(map_name)
            all_categories.update(by_cat.keys())

    for map_name in sorted(all_maps):
        for category in sorted(all_categories):
            any_data = any(
                map_x_category.get(cond, {}).get(map_name, {}).get(category)
                for cond in CONDITION_ORDER
            )
            if not any_data:
                continue
            print(f"Map: {map_name}   Category: {category}")
            header = (
                f"  {'Condition':<28} {'N':>4} "
                f"{'M1 (Task)':>12} {'M1 CI':>20} "
                f"{'M2 (Const)':>12} {'M3 (Subg)':>12}"
            )
            print(header)
            print("  " + "-" * 90)
            for cond in CONDITION_ORDER:
                cell = map_x_category.get(cond, {}).get(map_name, {}).get(category)
                if not cell:
                    continue
                name = CONDITION_NAMES.get(cond, cond)
                n = cell["n_episodes"]
                m1 = cell["M1_task_success"]
                m2 = cell["M2_constraint_adherence"]
                m3 = cell["M3_subgoal_success"]
                m1_str = f"{m1['rate']:.2%}" if m1["determined"] > 0 else "N/A"
                m1_ci = (
                    f"[{m1['wilson_ci_lower']:.3f}, {m1['wilson_ci_upper']:.3f}]"
                    if m1["determined"] > 0 else "N/A"
                )
                m2_str = f"{m2['rate']:.2%}" if m2["determined"] > 0 else "N/A"
                m3_str = f"{m3['rate']:.2%}" if m3["total_subgoals"] > 0 else "N/A"
                small_n = " (n<10)" if n < 10 else ""
                print(
                    f"  {name:<28} {n:>4} "
                    f"{m1_str:>12} {m1_ci:>20} "
                    f"{m2_str:>12} {m3_str:>12}{small_n}"
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


def print_subgoal_position_table(subgoal_position: dict[str, dict[int, dict]]):
    """Print subgoal success rates by ordinal position for each condition."""
    if not subgoal_position:
        return
    sep = "=" * 100
    print()
    print(sep)
    print(f"{'Subgoal Success by Position':^100}")
    print(sep)
    print()

    for cond in CONDITION_ORDER:
        pos_data = subgoal_position.get(cond)
        if not pos_data:
            continue
        name = CONDITION_NAMES.get(cond, cond)
        print(f"  {name}")
        print(f"    {'Position':>8} {'Success':>8} {'Total':>6} {'Rate':>8} {'95% CI':>22}")
        print("    " + "-" * 56)
        for pos_idx in sorted(pos_data.keys()):
            pd = pos_data[pos_idx]
            ci_str = f"[{pd['wilson_ci_lower']:.3f}, {pd['wilson_ci_upper']:.3f}]"
            print(
                f"    {pos_idx + 1:>8} {pd['successes']:>8} {pd['total']:>6} "
                f"{pd['rate']:.2%}{'':<2} {ci_str:>22}"
            )
        print()

    print(sep)
    print()


def write_json_output(
    condition_metrics: dict[str, dict],
    pairwise: dict[str, dict],
    output_path: Path,
    category_metrics: dict[str, dict[str, dict]] | None = None,
    map_metrics: dict[str, dict[str, dict]] | None = None,
    map_x_category: dict[str, dict[str, dict[str, dict]]] | None = None,
    mcnemar_results: dict[str, dict] | None = None,
    subgoal_position: dict[str, dict[int, dict]] | None = None,
):
    """Write full metrics to JSON."""
    output = {
        "metadata": {
            "script": "aggregate_results.py",
            "bonferroni_threshold": BONFERRONI_THRESHOLD,
            "bonferroni_family": "M1 only (6 C0-vs-baseline comparisons)",
            "m2_m3_significance": "exploratory (raw p-value at alpha=0.05; "
                                  "no family-wise correction claim)",
            "scipy_available": HAS_SCIPY,
        },
        "conditions": condition_metrics,
        "pairwise_comparisons": pairwise,
    }
    if category_metrics:
        output["category_breakdown"] = category_metrics
    if map_metrics:
        output["map_breakdown"] = map_metrics
    if map_x_category:
        output["map_x_category_breakdown"] = map_x_category
    if mcnemar_results:
        output["mcnemar_paired_tests"] = mcnemar_results
    if subgoal_position:
        output["subgoal_position_breakdown"] = subgoal_position
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON output written to: {output_path}")


def write_csv_output(
    condition_metrics: dict[str, dict],
    pairwise: dict[str, dict],
    output_path: Path,
    subgoal_position: dict[str, dict[int, dict]] | None = None,
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
        "m9_avg_cost_per_episode", "m9_total_cost",
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
            m9 = metrics["M9_api_cost"]

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
                "m9_avg_cost_per_episode": m9["avg_cost_per_episode"],
                "m9_total_cost": m9["total_cost"],
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

    # Write subgoal position CSV
    if subgoal_position:
        sp_path = output_path.parent / "subgoal_position.csv"
        sp_fields = ["condition", "position", "successes", "total", "rate"]
        with open(sp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sp_fields)
            writer.writeheader()
            for cond in CONDITION_ORDER:
                pos_data = subgoal_position.get(cond)
                if not pos_data:
                    continue
                for pos_idx in sorted(pos_data.keys()):
                    pd = pos_data[pos_idx]
                    writer.writerow({
                        "condition": cond,
                        "position": pos_idx,
                        "successes": pd["successes"],
                        "total": pd["total"],
                        "rate": pd["rate"],
                    })
        print(f"Subgoal position CSV written to: {sp_path}")

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
        help="Directory for output files (default: <results_dir>/aggregation)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific conditions (e.g. condition0 condition1)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any runs are missing required annotations "
             "OR if any condition has fewer than --expected-n completed runs.",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=EXPECTED_RUNS_PER_CONDITION,
        help=f"Expected completed runs per condition (default: "
             f"{EXPECTED_RUNS_PER_CONDITION} = 15 tasks x 3 variants). "
             "Set to 0 to disable the check.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or os.path.join(results_dir, "aggregation")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning results in: {os.path.abspath(results_dir)}")
    print()

    # Load all runs
    expected_n = args.expected_n if args.expected_n > 0 else None
    runs_by_condition = find_run_infos(
        results_dir, args.conditions,
        expected_n=expected_n, strict=args.strict,
    )

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

    # ---- Annotation validation ----
    # Three flavours of annotation are required for a fully reproducible run:
    #   (a) C1 and C3 task_success: manual video-review verdict (M1 source).
    #   (b) Every constrained-task episode (all 7 conditions) needs a
    #       manual ``any_constraint_violated`` flag in constraint_analysis.json
    #       so M2 is computed from the official source rather than the
    #       runtime monitor flag.
    #   (c) Each constrained task JSON should declare ``constraints_expected``
    #       so the annotator knows which constraints to look for.
    missing_annotations: list[dict] = []
    for cond in ["condition1", "condition3"]:
        for run in runs_by_condition.get(cond, []):
            ca = run.get("_constraint_analysis")
            if not ca or "task_success" not in ca:
                missing_annotations.append({
                    "condition": cond,
                    "path": run["_source_path"],
                    "missing": "task_success annotation in constraint_analysis.json",
                })
    for cond, runs in runs_by_condition.items():
        for run in runs:
            task = run.get("task", {})
            category = task.get("category")
            if category == "constrained":
                ce = task.get("constraints_expected")
                if not ce or not isinstance(ce, list) or len(ce) == 0:
                    missing_annotations.append({
                        "condition": cond,
                        "path": run["_source_path"],
                        "missing": "constraints_expected in task JSON",
                    })
                ca = run.get("_constraint_analysis")
                if not ca or "any_constraint_violated" not in ca:
                    missing_annotations.append({
                        "condition": cond,
                        "path": run["_source_path"],
                        "missing": "any_constraint_violated in constraint_analysis.json (manual M2 review)",
                    })

    if missing_annotations:
        print(f"WARNING: {len(missing_annotations)} runs require manual annotation "
              "before metrics can be fully computed.")
        for entry in missing_annotations:
            print(f"  [{entry['condition']}] {entry['path']}: {entry['missing']}")
        print()
        print("  - To annotate C1/C3 task success: add {\"task_success\": true/false} "
              "to constraint_analysis.json in each run directory")
        print("  - To annotate constraints: add constraints_expected list to the task JSON files")
        print()

        if args.strict:
            print("Exiting due to --strict flag.", file=sys.stderr)
            sys.exit(1)

    # Under --strict, M2 must come from the manual constraint_analysis.json
    # files; the runtime any_constraint_violated flag is suppressed. Missing
    # annotation cells will appear as "determined=0" rather than being
    # back-filled by the runtime monitor's flag.
    require_manual_m2 = bool(args.strict)

    # Compute per-condition metrics
    condition_metrics = {}
    for cond in CONDITION_ORDER:
        runs = runs_by_condition.get(cond, [])
        if runs:
            condition_metrics[cond] = aggregate_condition(
                cond, runs, require_manual_m2=require_manual_m2,
            )

    # Compute pairwise statistical tests
    pairwise = compute_pairwise_tests(condition_metrics)

    # Compute McNemar's test (paired by task_id)
    mcnemar_results = compute_paired_mcnemar_tests(runs_by_condition)

    # Compute per-category breakdown (sequential vs constrained)
    category_metrics = aggregate_by_category(
        runs_by_condition, require_manual_m2=require_manual_m2,
    )

    # Compute per-map breakdown (generalization across environments)
    map_metrics = aggregate_by_map(
        runs_by_condition, require_manual_m2=require_manual_m2,
    )

    # Compute per-map x per-category cross-tab (lead with this in the paper:
    # the marginal per-map and per-category tables are confounded because
    # task-category distribution is uneven across maps).
    map_x_category = aggregate_by_map_and_category(
        runs_by_condition, require_manual_m2=require_manual_m2,
    )

    # Compute subgoal position breakdown
    subgoal_position = aggregate_by_subgoal_position(runs_by_condition)

    # Output
    print_summary_table(condition_metrics, pairwise)
    print_mcnemar_table(mcnemar_results)
    print_category_table(category_metrics)
    print_map_table(map_metrics)
    print_map_x_category_table(map_x_category)
    print_subgoal_position_table(subgoal_position)

    # Write files
    output_path = Path(output_dir)
    write_json_output(
        condition_metrics, pairwise, output_path / "experiment_summary.json",
        category_metrics=category_metrics,
        map_metrics=map_metrics,
        map_x_category=map_x_category,
        mcnemar_results=mcnemar_results,
        subgoal_position=subgoal_position,
    )
    write_csv_output(
        condition_metrics, pairwise, output_path / "experiment_summary.csv",
        subgoal_position=subgoal_position,
    )


if __name__ == "__main__":
    main()
