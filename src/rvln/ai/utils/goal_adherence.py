"""
Goal adherence utilities: offline diary-based completion checking and
higher-level frame analysis helpers.

Low-level vision primitives (build_frame_grid, query_vlm, etc.) live in
vision.py; this module builds on them for goal-adherence-specific workflows
such as the offline ``check_subtask_completed_diary`` helper.

Prompt templates are defined in ``rvln.ai.prompts``.
"""

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from ...config import DEFAULT_VLM_MODEL
from ..prompts import (
    DEFAULT_TEMPORAL_PROMPT,
    DRONE_GOAL_MONITOR_CONTEXT,
    PROMPT_SUBTASK_COMPLETE,
    SUBTASK_COMPLETE_DIARY_PROMPT,
    WHAT_CHANGED_PROMPT,
)
from .vision import (
    build_frame_grid,
    get_ordered_frames_from_dir,
    query_vlm,
    sample_frames_every_n,
)

logger = logging.getLogger(__name__)


def build_prompt_what_changed(subtask: str) -> str:
    """Build the 'what changed' prompt: concise, one sentence, key facts only."""
    return WHAT_CHANGED_PROMPT.format(subtask=subtask)


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------

def analyze_temporal_frames(
    frame_paths: Union[Path, str, List[Path]],
    n: int,
    prompt: Union[str, None] = None,
    model: str = DEFAULT_VLM_MODEL,
    **grid_kwargs: Any,
) -> str:
    """Load frames, sample every n, build grid, query the VLM."""
    if isinstance(frame_paths, (Path, str)):
        frame_paths = get_ordered_frames_from_dir(Path(frame_paths))
    if not frame_paths:
        raise ValueError("No frames found (empty directory or list)")
    sampled = sample_frames_every_n(frame_paths, n)
    if not sampled:
        raise ValueError("No frames after sampling")
    grid = build_frame_grid(sampled, **grid_kwargs)
    return query_vlm(grid, prompt or DEFAULT_TEMPORAL_PROMPT, model=model)


def parse_yes_no_response(text: str) -> Optional[bool]:
    """Parse Yes/No subtask completion response."""
    if not text:
        return None
    t = text.strip().lower()
    if "yes the subtask is complete" in t or "yes, the subtask is complete" in t:
        return True
    if "no the subtask is not complete" in t or "no, the subtask is not complete" in t:
        return False
    if t.startswith("yes ") or t.startswith("yes.") or t == "yes":
        return True
    if t.startswith("no ") or t.startswith("no.") or t == "no":
        return False
    return None


def query_what_changed_between_frames(
    grid_image: Any,
    subtask: str,
    model: str = DEFAULT_VLM_MODEL,
    **kw: Any,
) -> str:
    """Ask what changed between the two frames in a 2-frame grid."""
    prompt = build_prompt_what_changed(subtask)
    return query_vlm(grid_image, prompt, model=model)


def check_subtask_complete_diary(
    frame_paths: Union[Path, str, List[Path]],
    subtask: str,
    n: int,
    artifacts_dir: Optional[Path] = None,
    model: str = DEFAULT_VLM_MODEL,
    **grid_kwargs: Any,
) -> dict:
    """Offline diary-based completion check over a sequence of saved frames."""
    def _frame_source() -> str:
        if isinstance(frame_paths, (Path, str)):
            return str(Path(frame_paths))
        return f"list of {len(frame_paths)} paths"

    if isinstance(frame_paths, (Path, str)):
        frames = get_ordered_frames_from_dir(Path(frame_paths))
    else:
        frames = list(frame_paths)

    result_empty: dict = {
        "complete": False,
        "frame_index": None,
        "diary": [],
        "reasoning": None,
        "last_checkpoint": None,
    }
    if not frames:
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            config = {"subtask": subtask, "n": n, "model": model,
                       "frame_source": _frame_source(), "total_frames": 0,
                       "grid_kwargs": grid_kwargs}
            (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))
            (artifacts_dir / "summary.json").write_text(json.dumps(result_empty, indent=2))
        return result_empty

    T_len = len(frames)
    if T_len < n + 1:
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            config = {"subtask": subtask, "n": n, "model": model,
                       "frame_source": _frame_source(), "total_frames": T_len,
                       "grid_kwargs": grid_kwargs}
            (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))
            (artifacts_dir / "summary.json").write_text(json.dumps(result_empty, indent=2))
        return result_empty

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config = {"subtask": subtask, "n": n, "model": model,
                   "frame_source": _frame_source(), "total_frames": T_len,
                   "grid_kwargs": grid_kwargs}
        (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))

    diary: List[str] = []
    last_checkpoint: Optional[int] = None

    k = 1
    while k * n < T_len:
        current = k * n
        last_checkpoint = current

        two_paths = [frames[current - n], frames[current]]
        grid_two = build_frame_grid(two_paths, **grid_kwargs)
        prompt_wc = build_prompt_what_changed(subtask)
        change_text = query_what_changed_between_frames(
            grid_two, subtask, model=model
        )
        diary.append(f"Between frame {current - n} and frame {current}: {change_text}")

        if artifacts_dir is not None:
            cp_dir = artifacts_dir / f"checkpoint_{current}"
            cp_dir.mkdir(parents=True, exist_ok=True)
            grid_two.save(cp_dir / "grid_two_frames.png")
            (cp_dir / "prompt_what_changed.txt").write_text(prompt_wc)
            (cp_dir / "response_what_changed.txt").write_text(change_text)

        capped = frames[: current + 1]
        sampled = sample_frames_every_n(capped, n)
        if not sampled:
            k += 1
            continue
        grid_full = build_frame_grid(sampled, **grid_kwargs)
        diary_blob = "\n".join(diary)
        prompt_b = SUBTASK_COMPLETE_DIARY_PROMPT.format(
            subtask=subtask, diary_blob=diary_blob,
        )
        response_b = query_vlm(grid_full, prompt_b, model=model)

        if artifacts_dir is not None:
            cp_dir = artifacts_dir / f"checkpoint_{current}"
            cp_dir.mkdir(parents=True, exist_ok=True)
            grid_full.save(cp_dir / "grid_full.png")
            (cp_dir / "prompt_complete.txt").write_text(prompt_b)
            (cp_dir / "response_complete.txt").write_text(response_b)

        complete = parse_yes_no_response(response_b)
        if complete is True:
            result = {
                "complete": True,
                "frame_index": current,
                "diary": list(diary),
                "reasoning": response_b,
                "last_checkpoint": current,
            }
            if artifacts_dir is not None:
                (artifacts_dir / "summary.json").write_text(json.dumps(result, indent=2))
            return result
        k += 1

    result = {
        "complete": False,
        "frame_index": None,
        "diary": list(diary),
        "reasoning": None,
        "last_checkpoint": last_checkpoint,
    }
    if artifacts_dir is not None:
        (artifacts_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result
