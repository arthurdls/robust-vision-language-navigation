"""
Goal adherence approach 2.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from goal_adherence_utils import (
    analyze_temporal_frames,
    build_frame_grid,
    check_subtask_complete_fewshot,
    check_subtask_complete_fewshot_checkpoints,
    get_ordered_frames_from_dir,
    query_gpt4o_mini_temporal,
    sample_frames_every_n,
)

__all__ = [
    "sample_frames_every_n",
    "get_ordered_frames_from_dir",
    "build_frame_grid",
    "query_gpt4o_mini_temporal",
    "analyze_temporal_frames",
    "check_subtask_complete_fewshot",
    "check_subtask_complete_fewshot_checkpoints",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build frame grid and query gpt-4o-mini for temporal analysis."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Results directory (REPL or LTL frame layout).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Sample every N frames (default: 10).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt; default asks what is happening across the frames.",
    )
    parser.add_argument(
        "--grid-only",
        type=Path,
        metavar="OUTPUT.png",
        default=None,
        help="Build the grid and save to this path (no API call). Use - to show in viewer.",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default=None,
        help="Subtask in natural language; requires --examples for few-shot completion check.",
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=None,
        metavar="JSON",
        help="Path to JSON file: [{\"path\": \"...\", \"complete_at\": 45 or null}, ...].",
    )
    parser.add_argument(
        "--save-artifacts",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root directory for artifacts (default: goal_adherence_artifacts in cwd).",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Disable saving artifacts for this run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model (default: gpt-4o-mini; use gpt-4o for full model).",
    )
    args = parser.parse_args()
    if args.grid_only is not None:
        paths = get_ordered_frames_from_dir(args.dir)
        if not paths:
            print("No frames found.", file=sys.stderr)
            sys.exit(1)
        sampled = sample_frames_every_n(paths, args.n)
        grid = build_frame_grid(sampled)
        if str(args.grid_only) == "-":
            fd, path = tempfile.mkstemp(suffix=".png", prefix="goal_grid_")
            os.close(fd)
            grid.save(path)
            try:
                if sys.platform == "darwin":
                    subprocess.run(["open", path], check=True)
                elif sys.platform == "win32":
                    os.startfile(path)
                else:
                    subprocess.run(["xdg-open", path], check=True)
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"Could not open viewer: {e}. Grid saved to: {path}", file=sys.stderr)
            else:
                print("Opened grid (saved to", path + ")")
        else:
            grid.save(args.grid_only)
            print("Saved grid to", args.grid_only)
        sys.exit(0)
    if args.subtask is not None and args.examples is not None:
        with open(args.examples) as f:
            raw = json.load(f)
        few_shot_examples: List[tuple] = []
        for item in raw:
            path = Path(item["path"])
            complete_at = item.get("complete_at")
            if complete_at is not None and not isinstance(complete_at, int):
                complete_at = int(complete_at)
            example_subtask = item.get("subtask")
            few_shot_examples.append((path, complete_at, example_subtask))
        if args.no_save_artifacts:
            run_dir = None
        else:
            root = args.save_artifacts if args.save_artifacts is not None else Path.cwd() / "goal_adherence_artifacts"
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            run_dir = root / f"run_{timestamp}"
        result = check_subtask_complete_fewshot_checkpoints(
            args.dir, args.subtask, few_shot_examples, args.n, artifacts_dir=run_dir, model=args.model
        )
        if run_dir is not None:
            print("Artifacts saved to", run_dir)
        if result["complete"]:
            print(f"Complete at frame {result['frame_index']}")
            if result.get("reasoning"):
                print("Reasoning:", result["reasoning"])
        else:
            last = result.get("last_checkpoint")
            if last is not None:
                print(f"Not complete by end (last checkpoint frame {last})")
            else:
                print("Not complete (no checkpoints or no frames).")
            if result.get("reasoning"):
                print("Reasoning:", result["reasoning"])
        sys.exit(0)
    response = analyze_temporal_frames(args.dir, args.n, prompt=args.prompt, model=args.model)
    print(response)
