#!/usr/bin/env python3
"""
Post-hoc constraint violation analysis for completed experiment runs.

Passes sampled trajectory frames through a VLM to detect constraint violations.
This is especially needed for conditions C1 and C3 which have no runtime constraint
monitoring, but can also re-analyze any condition's results for consistency.

Usage (single run):
  python scripts/analyze_constraint_adherence.py \
    results/condition1/c1_naive__seq_constraint_01__2026_04_30_05_08_40

Usage (batch, all runs in a directory):
  python scripts/analyze_constraint_adherence.py \
    --results_dir results/condition1

Options:
  --model          VLM model to use (default: gpt-5.4)
  --sample_every   Frame sampling interval (default: 20)
  --force          Re-analyze even if constraint_analysis.json already exists
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rvln.ai.utils.llm_providers import LLMFactory
from rvln.ai.utils.vision import (
    build_frame_grid,
    get_ordered_frames_from_dir,
    query_vlm,
    sample_frames_every_n,
)
from rvln.config import DEFAULT_VLM_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint extraction
# ---------------------------------------------------------------------------

def extract_constraints_from_run(run_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract constraint descriptions from run_info.json.

    Checks multiple sources in priority order:
      1. task.constraints_expected (if present in the task dict)
      2. subgoal_summaries[*].constraints (collected from runtime monitoring)
      3. Falls back to deriving ordering constraints from the task instruction

    Returns a list of dicts with keys: description, polarity.
    """
    task = run_info.get("task", {})

    # Source 1: explicit constraints_expected in the task definition
    constraints_expected = task.get("constraints_expected")
    if constraints_expected and isinstance(constraints_expected, list):
        return constraints_expected

    # Source 2: collect unique constraints from subgoal summaries
    seen = set()
    collected = []
    for sg in run_info.get("subgoal_summaries", []):
        for c in sg.get("constraints", []):
            desc = c if isinstance(c, str) else c.get("description", "")
            if desc and desc not in seen:
                seen.add(desc)
                polarity = "negative" if isinstance(c, dict) else "negative"
                if isinstance(c, dict):
                    polarity = c.get("polarity", "negative")
                collected.append({"description": desc, "polarity": polarity})
    if collected:
        return collected

    # Source 3: derive ordering constraints from instruction
    # For sequential tasks, the implicit constraint is that subgoals must be
    # visited in order (i.e., do not skip ahead or revisit).
    instruction = task.get("instruction", "")
    if not instruction:
        instruction = run_info.get("instruction_sent", "")

    return _derive_constraints_from_instruction(instruction)


def _derive_constraints_from_instruction(instruction: str) -> List[Dict[str, str]]:
    """Derive implicit ordering/negative constraints from the task instruction.

    Looks for phrases like 'stay away from', 'avoid', 'do not', etc.
    Also extracts ordering constraints from sequential instructions.
    """
    constraints = []

    # Look for explicit negative constraint language
    negative_phrases = [
        "stay away from",
        "avoid",
        "do not go near",
        "don't go near",
        "do not touch",
        "don't touch",
        "do not pass through",
        "don't pass through",
        "keep away from",
        "never",
    ]
    instruction_lower = instruction.lower()
    for phrase in negative_phrases:
        idx = instruction_lower.find(phrase)
        if idx != -1:
            # Extract the rest of the clause (until next comma, period, or 'then')
            rest = instruction[idx:]
            for delim in [",", ".", " then ", " and then "]:
                delim_idx = rest.lower().find(delim)
                if delim_idx > 0:
                    rest = rest[:delim_idx]
                    break
            constraints.append({
                "description": rest.strip(),
                "polarity": "negative",
            })

    # If the task is sequential (has 'then'), add an ordering constraint
    if " then " in instruction_lower:
        constraints.append({
            "description": "subgoals must be completed in sequential order",
            "polarity": "positive",
        })

    return constraints


# ---------------------------------------------------------------------------
# VLM constraint analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a constraint-violation detector for a UAV navigation experiment.
You will be shown a grid of frames sampled from the UAV's camera feed during
a navigation task. Your job is to determine whether a specific constraint was
violated at any point during the trajectory shown.

Analyze the frames carefully. The frames are shown in temporal order, left to
right, top to bottom. Each frame represents a moment in the trajectory.

Respond ONLY with valid JSON (no markdown fences). Use this exact schema:
{
  "violated": true or false,
  "violation_frames": [list of 0-indexed frame numbers where violation is visible],
  "confidence": "high" or "medium" or "low",
  "reasoning": "brief explanation of your determination"
}
"""


def build_constraint_prompt(
    constraint: Dict[str, str],
    instruction: str,
    num_frames: int,
) -> str:
    """Build the user-role prompt for a single constraint check."""
    polarity = constraint.get("polarity", "negative")
    desc = constraint["description"]

    if polarity == "negative":
        question = (
            f"The UAV was given this navigation instruction:\n"
            f'"{instruction}"\n\n'
            f"The constraint to check is (NEGATIVE, meaning it should NOT happen):\n"
            f'"{desc}"\n\n'
            f"Looking at these {num_frames} frames from the trajectory, "
            f"was this negative constraint violated? In other words, did the "
            f"prohibited action or situation occur at any point?\n\n"
            f"Respond with JSON only."
        )
    else:
        question = (
            f"The UAV was given this navigation instruction:\n"
            f'"{instruction}"\n\n'
            f"The constraint to check is (POSITIVE, meaning it SHOULD be maintained):\n"
            f'"{desc}"\n\n'
            f"Looking at these {num_frames} frames from the trajectory, "
            f"was this positive constraint violated? In other words, was the "
            f"required condition NOT maintained at any point?\n\n"
            f"Respond with JSON only."
        )
    return question


def analyze_constraint(
    constraint: Dict[str, str],
    instruction: str,
    grid_image: Any,
    num_frames: int,
    model: str,
    llm: Any,
) -> Dict[str, Any]:
    """Analyze a single constraint against the frame grid via VLM.

    Returns a dict with: description, polarity, violated, violation_frames,
    confidence, reasoning.
    """
    prompt = build_constraint_prompt(constraint, instruction, num_frames)

    try:
        response_text = query_vlm(
            grid_image=grid_image,
            prompt=prompt,
            model=model,
            llm=llm,
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error("VLM query failed for constraint '%s': %s", constraint["description"], e)
        return {
            "description": constraint["description"],
            "polarity": constraint.get("polarity", "negative"),
            "violated": None,
            "violation_frames": [],
            "confidence": "low",
            "reasoning": f"VLM query failed: {e}",
        }

    # Parse VLM response
    result = _parse_vlm_response(response_text)
    result["description"] = constraint["description"]
    result["polarity"] = constraint.get("polarity", "negative")
    return result


def _parse_vlm_response(response_text: str) -> Dict[str, Any]:
    """Parse the VLM JSON response, handling common formatting issues."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        return {
            "violated": bool(data.get("violated", False)),
            "violation_frames": data.get("violation_frames", []),
            "confidence": data.get("confidence", "medium"),
            "reasoning": data.get("reasoning", ""),
        }
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse VLM response as JSON: %s", e)
        logger.debug("Raw response: %s", text)
        # Try to extract key information from free-text
        violated = any(
            kw in text.lower()
            for kw in ["violated: true", '"violated": true', "was violated", "violation detected"]
        )
        return {
            "violated": violated,
            "violation_frames": [],
            "confidence": "low",
            "reasoning": f"(parse failure) Raw response: {text[:500]}",
        }


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

def analyze_single_run(
    run_dir: Path,
    model: str = DEFAULT_VLM_MODEL,
    sample_every: int = 20,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Analyze constraint adherence for a single experiment run.

    Args:
        run_dir: Path to the run directory containing run_info.json and frames/
        model: VLM model name
        sample_every: sample every Nth frame
        force: re-analyze even if constraint_analysis.json already exists

    Returns:
        The analysis result dict, or None if skipped.
    """
    run_dir = Path(run_dir)
    output_path = run_dir / "constraint_analysis.json"

    if output_path.exists() and not force:
        logger.info("Skipping %s (constraint_analysis.json exists, use --force to re-run)", run_dir.name)
        return None

    # Load run_info.json
    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        logger.warning("No run_info.json found in %s, skipping", run_dir)
        return None

    with open(run_info_path) as f:
        run_info = json.load(f)

    # Get task info
    task = run_info.get("task", {})
    task_id = task.get("task_id", run_dir.name)
    condition = run_info.get("condition", "unknown")
    instruction = task.get("instruction", "")
    if not instruction:
        instruction = run_info.get("instruction_sent", "")

    # Extract constraints to analyze
    constraints = extract_constraints_from_run(run_info)
    if not constraints:
        logger.info("No constraints found for %s, skipping analysis", run_dir.name)
        # Still produce an output file noting no constraints
        result = {
            "task_id": task_id,
            "condition": condition,
            "constraints_analyzed": [],
            "any_constraint_violated": False,
            "analysis_model": model,
            "frames_sampled": 0,
            "total_frames": 0,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "No constraints found to analyze",
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote %s (no constraints)", output_path)
        return result

    # Get frames
    all_frames = get_ordered_frames_from_dir(run_dir)
    total_frames = len(all_frames)
    if total_frames == 0:
        logger.warning("No frames found in %s, skipping", run_dir)
        return None

    # Sample frames
    sampled_frames = sample_frames_every_n(all_frames, sample_every)
    num_sampled = len(sampled_frames)
    logger.info(
        "Analyzing %s: %d constraints, %d/%d frames sampled (every %d)",
        run_dir.name, len(constraints), num_sampled, total_frames, sample_every,
    )

    # Build frame grid
    grid_image = build_frame_grid(sampled_frames)

    # Create LLM instance (reuse across constraints)
    if model.startswith("gemini"):
        llm = LLMFactory.create("gemini", model=model)
    else:
        llm = LLMFactory.create("openai", model=model)

    # Analyze each constraint
    constraint_results = []
    for i, constraint in enumerate(constraints):
        logger.info(
            "  Constraint %d/%d: %s", i + 1, len(constraints), constraint["description"]
        )
        result = analyze_constraint(
            constraint=constraint,
            instruction=instruction,
            grid_image=grid_image,
            num_frames=num_sampled,
            model=model,
            llm=llm,
        )
        constraint_results.append(result)
        # Brief pause between VLM calls to be respectful of rate limits
        if i < len(constraints) - 1:
            time.sleep(0.5)

    # Determine overall violation status
    any_violated = any(
        cr.get("violated") is True for cr in constraint_results
    )

    # Build output
    output = {
        "task_id": task_id,
        "condition": condition,
        "constraints_analyzed": constraint_results,
        "any_constraint_violated": any_violated,
        "analysis_model": model,
        "frames_sampled": num_sampled,
        "total_frames": total_frames,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote %s", output_path)

    return output


# ---------------------------------------------------------------------------
# Batch mode: discover runs in a directory
# ---------------------------------------------------------------------------

def discover_runs(results_dir: Path) -> List[Path]:
    """Find all run directories under the given path.

    A run directory is identified by the presence of run_info.json.
    Searches up to 3 levels deep to handle both flat and map-nested layouts:
      results/conditionN/<run_name>/run_info.json
      results/conditionN/<map_name>/<run_name>/run_info.json
    """
    results_dir = Path(results_dir)
    runs = []

    # Direct: results_dir itself is a run
    if (results_dir / "run_info.json").exists():
        return [results_dir]

    # Search for run_info.json up to 3 levels deep
    for depth_pattern in ["*/run_info.json", "*/*/run_info.json", "*/*/*/run_info.json"]:
        for ri in sorted(results_dir.glob(depth_pattern)):
            runs.append(ri.parent)

    # Deduplicate (in case nested patterns overlap)
    seen = set()
    unique_runs = []
    for r in runs:
        rr = r.resolve()
        if rr not in seen:
            seen.add(rr)
            unique_runs.append(r)

    return unique_runs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc constraint violation analysis for experiment runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Analyze a single run:\n"
            "  python scripts/analyze_constraint_adherence.py results/condition1/c1_naive__seq_constraint_01__...\n"
            "\n"
            "  # Analyze all runs in a condition directory:\n"
            "  python scripts/analyze_constraint_adherence.py --results_dir results/condition1\n"
            "\n"
            "  # Use a different model and sampling rate:\n"
            "  python scripts/analyze_constraint_adherence.py --results_dir results/ --model gpt-4o --sample_every 10\n"
        ),
    )
    parser.add_argument(
        "run_path",
        nargs="?",
        type=Path,
        help="Path to a single run directory to analyze",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        help="Path to a results directory (batch mode: analyze all runs found within)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_VLM_MODEL,
        help=f"VLM model to use for analysis (default: {DEFAULT_VLM_MODEL})",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=20,
        help="Frame sampling interval, every Nth frame (default: 20)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze even if constraint_analysis.json already exists",
    )

    args = parser.parse_args()

    # Determine which runs to analyze
    if args.run_path and args.results_dir:
        parser.error("Specify either a single run_path or --results_dir, not both")

    if args.run_path:
        runs = discover_runs(args.run_path)
    elif args.results_dir:
        runs = discover_runs(args.results_dir)
    else:
        parser.error("Provide a run_path or --results_dir")

    if not runs:
        logger.error("No runs found to analyze.")
        sys.exit(1)

    logger.info("Found %d run(s) to analyze", len(runs))

    # Process each run
    results = []
    for run_dir in runs:
        try:
            result = analyze_single_run(
                run_dir=run_dir,
                model=args.model,
                sample_every=args.sample_every,
                force=args.force,
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error("Failed to analyze %s: %s", run_dir, e, exc_info=True)

    # Summary
    analyzed = len(results)
    violated = sum(1 for r in results if r.get("any_constraint_violated"))
    logger.info(
        "Analysis complete: %d runs analyzed, %d with constraint violations",
        analyzed, violated,
    )


if __name__ == "__main__":
    main()
