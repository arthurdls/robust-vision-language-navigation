"""
Frame playback and MP4 encoding utilities for experiment results.

Provides functions for discovering frames in a run directory and encoding
them to video. Used by scripts/playback.py and experiment runners.
"""

import json
import logging

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


def load_frame_labels(run_dir: Path) -> Dict[int, str]:
    """Build a mapping from primary-timeline frame index to subgoal/command label.

    Reads ``run_info.json`` and uses ``subgoal_summaries`` (LTL / integration)
    or ``instruction_overrides`` (goal adherence) to determine which label
    applies to each frame.

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

    labels: Dict[int, str] = {}

    # --- LTL / integration: subgoal_summaries with total_steps ---
    summaries = info.get("subgoal_summaries")
    if summaries and all("total_steps" in s for s in summaries):
        offset = 0
        for s in summaries:
            label = s.get("subgoal", s.get("converted_instruction", ""))
            steps = s["total_steps"]
            for i in range(steps):
                labels[offset + i] = label
            offset += steps
        return labels

    # --- Goal adherence: single instruction with optional overrides ---
    instruction = info.get("instruction_sent")
    if isinstance(instruction, str) and instruction:
        total = info.get("total_steps")
        overrides = info.get("instruction_overrides", [])
        if total:
            current = instruction
            override_idx = 0
            for i in range(total):
                while override_idx < len(overrides) and i >= overrides[override_idx].get("step", total):
                    current = overrides[override_idx].get("new_instruction", current)
                    override_idx += 1
                labels[i] = current
            return labels

    return labels



def draw_label_overlay(img, text: str) -> None:
    """Draw *text* on the top-left of *img* (BGR numpy array, modified in place).

    Uses a semi-transparent black background strip with white text for
    readability over any frame content.
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    color = (255, 255, 255)
    margin = 8

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    max_w = img.shape[1] - 2 * margin
    if tw > max_w:
        scale = scale * max_w / tw
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = margin, margin + th
    pad = 4
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        cv2.FILLED,
    )
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)



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
    frame_labels: Optional[Dict[int, str]] = None,
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
        Optional mapping from frame index to overlay label text.
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
