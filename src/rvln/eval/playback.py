"""
Frame playback and MP4 encoding utilities for experiment results.

Provides functions for discovering frames in a run directory and encoding
them to video. Used by scripts/playback.py and experiment runners.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


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
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("Skipping unreadable frame: %s", p)
                continue
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
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
) -> Optional[Path]:
    """Write ``run_dir / out_name`` from frames under *run_dir*.

    Returns the output path, or ``None`` if no frames were found.
    """
    run_dir = Path(run_dir)
    paths = iter_run_frame_paths(run_dir)
    if not paths:
        logger.warning("No frames found under %s; skipping MP4.", run_dir)
        return None
    out_path = run_dir / out_name
    write_frames_to_mp4(paths, out_path, fps=fps, fourcc=fourcc)
    logger.info("Saved video to %s", out_path)
    return out_path
