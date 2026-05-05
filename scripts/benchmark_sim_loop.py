"""Benchmark a fixed number of control-loop iterations and print stats.

Usage:
    python scripts/benchmark_sim_loop.py path/to/step_timings.jsonl

Reads step_timings.jsonl produced by the runner and prints median/p95
per phase. Used to verify each speedup task lands.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(path: Path) -> None:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not records:
        print(f"No records in {path}")
        return
    keys = sorted({k for r in records for k in r if k.endswith("_ms")})
    print(f"{path} ({len(records)} steps)")
    print(f"{'phase':<24} {'median':>8} {'p95':>8} {'max':>8}")
    for k in keys:
        vals = [r.get(k, 0.0) for r in records]
        p95 = statistics.quantiles(vals, n=20)[-1] if len(vals) >= 2 else max(vals)
        print(f"{k:<24} {statistics.median(vals):>8.1f} "
              f"{p95:>8.1f} "
              f"{max(vals):>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path,
                        help="Path to step_timings.jsonl")
    args = parser.parse_args()
    summarize(args.path)


if __name__ == "__main__":
    main()
