"""
Frame playback and MP4 encoding utilities for experiment results.

Provides functions for discovering frames in a run directory and encoding
them to video. Used by scripts/playback.py and experiment runners.
"""

import json
import logging

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# A per-frame label is a (subgoal, openvla_command) pair. Either may be empty
# when not applicable; an empty string causes the corresponding overlay strip
# to be skipped.
FrameLabel = Tuple[str, str]


def load_frame_labels(run_dir: Path) -> Dict[int, FrameLabel]:
    """Build a mapping from primary-timeline frame index to (subgoal, command).

    Reads ``run_info.json`` and produces, for each frame index, the active
    sub-task label (top overlay) and the low-level OpenVLA command (bottom
    overlay) that the policy was conditioned on at that frame.

    Sources:
      * LTL / integration / ablations: ``subgoal_summaries`` provides per-subgoal
        ``total_steps``, ``subgoal``, ``converted_instruction``, and
        ``override_history`` (mid-subgoal command corrections, step-indexed
        within the subgoal).
      * Naive (Condition 1): ``instruction_sent`` plus optional
        ``instruction_overrides`` (step-indexed across the whole run).

    The returned dict is then extended (carry-forward) so every primary-
    timeline frame on disk has a label, even when ``total_steps``
    undercounts (e.g., the monitor returns stop/ask_help at the start of a
    step after the frame for that step was already written to disk).

    Returns an empty dict when no usable metadata is found.
    """
    run_dir = Path(run_dir)
    info_path = run_dir / "run_info.json"
    if not info_path.exists():
        return {}

    try:
        info = json.loads(info_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    labels: Dict[int, FrameLabel] = {}

    # --- LTL / integration / ablations: subgoal_summaries with total_steps ---
    summaries = info.get("subgoal_summaries")
    if summaries and all("total_steps" in s for s in summaries):
        offset = 0
        for s in summaries:
            subgoal = s.get("subgoal", "")
            converted = s.get("converted_instruction", "") or subgoal
            steps = s["total_steps"]
            override_history = s.get("override_history", []) or []
            # Sort by within-subgoal step so we can scan linearly.
            sorted_overrides = sorted(
                (o for o in override_history if isinstance(o, dict) and "step" in o),
                key=lambda o: o["step"],
            )
            current_subgoal = subgoal
            current_command = converted
            override_idx = 0
            for i in range(steps):
                while (
                    override_idx < len(sorted_overrides)
                    and sorted_overrides[override_idx]["step"] <= i
                ):
                    o = sorted_overrides[override_idx]
                    new_instr = o.get("new_instruction")
                    if isinstance(new_instr, str) and new_instr:
                        current_command = new_instr
                    new_subgoal = o.get("new_subgoal")
                    if isinstance(new_subgoal, str) and new_subgoal:
                        current_subgoal = new_subgoal
                    override_idx += 1
                labels[offset + i] = (current_subgoal, current_command)
            offset += steps
        _extend_labels_to_disk(labels, run_dir)
        return labels

    # --- Naive: single instruction with optional overrides ---
    instruction = info.get("instruction_sent")
    if isinstance(instruction, str) and instruction:
        total = info.get("total_steps")
        overrides = info.get("instruction_overrides", []) or []
        if total:
            current = instruction
            override_idx = 0
            for i in range(total):
                while (
                    override_idx < len(overrides)
                    and i >= overrides[override_idx].get("step", total)
                ):
                    current = overrides[override_idx].get("new_instruction", current)
                    override_idx += 1
                # No subgoal decomposition in naive runs: top and bottom are
                # the same instruction.
                labels[i] = (current, current)
            _extend_labels_to_disk(labels, run_dir)
            return labels

    return labels


def _extend_labels_to_disk(labels: Dict[int, FrameLabel], run_dir: Path) -> None:
    """Carry the most recent label forward to cover every frame on disk.

    The runner can write a frame for step ``S`` and then set
    ``total_steps = S`` when the monitor returns stop/ask_help at the start
    of the step (no action applied). For the final subgoal this leaves a
    trailing frame on disk with no entry in *labels*; for non-final
    subgoals the trailing frame is overwritten by the next subgoal's first
    frame and is not visible here. Filling forward ensures every
    primary-timeline frame still has a (subgoal, command) overlay.
    """
    if not labels:
        return
    paths = iter_run_frame_paths(run_dir)
    if not paths:
        return
    last = labels[max(labels)]
    for i in range(len(paths)):
        if i in labels:
            last = labels[i]
        else:
            labels[i] = last



def _draw_text_strip(img, text: str, *, position: str) -> None:
    """Draw a single semi-transparent text strip at top or bottom of *img*.

    *position* must be "top" or "bottom". Empty *text* is a no-op.
    """
    if not text:
        return

    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    color = (255, 255, 255)
    margin = 8
    pad = 4
    alpha = 0.7

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    max_w = img.shape[1] - 2 * margin
    if tw > max_w:
        scale = scale * max_w / tw
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    if position == "top":
        x = margin
        y = margin + th
    elif position == "bottom":
        x = margin
        y = img.shape[0] - margin - baseline
    else:
        raise ValueError(f"position must be 'top' or 'bottom', got {position!r}")

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_label_overlay(
    img,
    label: Union[str, FrameLabel],
    *,
    bottom: Optional[str] = None,
) -> None:
    """Draw subgoal (top) and OpenVLA command (bottom) overlays on *img*.

    *label* accepts either:
      * a (subgoal, command) tuple -- subgoal goes on top, command on bottom
      * a single string (legacy) -- drawn on top only, unless *bottom* is given

    Empty strings skip the corresponding strip. *img* is BGR and modified
    in place.
    """
    if isinstance(label, tuple):
        top_text, bottom_text = label
    else:
        top_text = label
        bottom_text = bottom or ""

    _draw_text_strip(img, top_text, position="top")
    _draw_text_strip(img, bottom_text, position="bottom")



def is_primary_timeline_frame(path: Union[str, Path]) -> bool:
    """True if *path* is a main-sequence frame (excludes convergence snapshots)."""
    name = Path(path).name
    return not name.startswith("frame_conv_")


def iter_run_frame_paths(run_dir: Path) -> List[Path]:
    """Return sorted PNG paths for a run directory.

    Prefers ``run_dir/frames/*.png`` (goal adherence, LTL), else ``run_dir/frame_*.png`` (REPL).
    Omits ``frame_conv_*.png`` (convergence snapshots; not part of the step timeline).
    """
    run_dir = Path(run_dir)
    frames_sub = run_dir / "frames"
    if frames_sub.is_dir():
        paths = sorted(frames_sub.glob("*.png"))
        return [p for p in paths if is_primary_timeline_frame(p)]
    repl = sorted(run_dir.glob("frame_*.png"))
    return [p for p in repl if is_primary_timeline_frame(p)]


def write_frames_to_mp4(
    frame_paths: Sequence[Union[str, Path]],
    out_path: Path,
    *,
    fps: float = 10.0,
    fourcc: str = "mp4v",
    frame_labels: Optional[Dict[int, Union[FrameLabel, str]]] = None,
) -> Path:
    """Encode image sequence to MP4 using OpenCV.

    Parameters
    ----------
    frame_paths
        Ordered list of image paths (typically PNG).
    out_path
        Output .mp4 path (parent directory is created if needed).
    fps
        Frame rate.
    fourcc
        Four-character codec.
    frame_labels
        Optional mapping from frame index to overlay label. Each value is a
        ``(subgoal, openvla_command)`` tuple drawn at the top and bottom of
        the frame respectively. Plain strings are also accepted for backward
        compatibility (drawn at the top only).
    """
    import cv2

    paths = [Path(p) for p in frame_paths]
    if not paths:
        raise ValueError("frame_paths is empty")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(paths[0]))
    if first is None:
        raise OSError(f"could not read first frame: {paths[0]}")

    h, w = first.shape[:2]
    cc = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(out_path), cc, float(fps), (w, h))
    if not writer.isOpened():
        writer.release()
        raise OSError(f"VideoWriter could not open {out_path} (codec={fourcc!r})")

    try:
        for idx, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("Skipping unreadable frame: %s", p)
                continue
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            if frame_labels and idx in frame_labels:
                draw_label_overlay(img, frame_labels[idx])
            writer.write(img)
    finally:
        writer.release()

    return out_path


def save_run_directory_mp4(
    run_dir: Path,
    *,
    out_name: str = "playback.mp4",
    fps: float = 10.0,
    fourcc: str = "mp4v",
    overlay: bool = True,
) -> Optional[Path]:
    """Write ``run_dir / out_name`` from frames under *run_dir*.

    Returns the output path, or ``None`` if no frames were found.
    """
    run_dir = Path(run_dir)
    paths = iter_run_frame_paths(run_dir)
    if not paths:
        logger.warning("No frames found under %s; skipping MP4.", run_dir)
        return None
    frame_labels = load_frame_labels(run_dir) if overlay else None
    out_path = run_dir / out_name
    write_frames_to_mp4(
        paths, out_path, fps=fps, fourcc=fourcc,
        frame_labels=frame_labels,
    )
    logger.info("Saved video to %s", out_path)
    return out_path
