"""
Vision utilities for frame sampling, grid building, and VLM image queries.

All VLM calls go through LLMFactory / BaseLLM so the same code works with
any configured provider (OpenAI, Gemini, etc.).
"""

import math
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, TypeVar, Union

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]

from .llm_providers import BaseLLM, LLMFactory

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def sample_frames_every_n(frames: Sequence[T], n: int) -> List[T]:
    """Return the last frame and every n-th frame backward, in temporal order.

    Example: 144 frames, n=10 -> indices [3, 13, 23, ..., 133, 143].
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not frames:
        return []
    last = len(frames) - 1
    indices = sorted(
        last - k * n for k in range(0, (last // n) + 1) if last - k * n >= 0
    )
    return [frames[i] for i in indices]


def get_ordered_frames_from_dir(dir_path: Path) -> List[Path]:
    """Discover frame paths in a results directory (LTL or REPL layout).

    Returns paths in temporal order.
    - LTL layout:  dir_path/frames/*.png
    - REPL layout: dir_path/frame_*.png
    """
    dir_path = Path(dir_path)
    frames_dir = dir_path / "frames"
    if frames_dir.is_dir():
        paths = sorted(frames_dir.glob("*.png"))
        if paths:
            return list(paths)
    repl_frames = sorted(dir_path.glob("frame_*.png"))
    if repl_frames:
        return list(repl_frames)
    return []


# ---------------------------------------------------------------------------
# Grid image construction
# ---------------------------------------------------------------------------

def build_frame_grid(
    frame_paths: List[Path],
    padding_px: int = 16,
    max_size: int = 2048,
    min_short_side: int = 768,
) -> "Image.Image":
    """Build a single image grid from ordered frame paths with white padding.

    Formula: width  = cols * frame_w + (cols + 1) * padding_px
             height = rows * frame_h + (rows + 1) * padding_px
    """
    if Image is None:
        raise RuntimeError("PIL is required. Install with: pip install Pillow")
    if not frame_paths:
        raise ValueError("frame_paths must not be empty")

    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        frames.append(img)

    K = len(frames)
    cols = math.ceil(math.sqrt(K))
    rows = math.ceil(K / cols)
    p = padding_px

    cell_w, cell_h = frames[0].size
    cell_w = max(cell_w, 1)
    cell_h = max(cell_h, 1)

    W = int(cols * cell_w + (cols + 1) * p)
    H = int(rows * cell_h + (rows + 1) * p)

    grid = Image.new("RGB", (W, H), (255, 255, 255))

    for idx, im in enumerate(frames):
        r = idx // cols
        c = idx % cols
        im_w, im_h = im.size
        scale = min(cell_w / im_w, cell_h / im_h, 1.0) if im_w and im_h else 1.0
        new_w = max(1, int(im_w * scale))
        new_h = max(1, int(im_h * scale))
        im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x = p + c * (cell_w + p) + (cell_w - new_w) // 2
        y = p + r * (cell_h + p) + (cell_h - new_h) // 2
        grid.paste(im_resized, (x, y))

    return grid


# ---------------------------------------------------------------------------
# VLM image queries (provider-agnostic via BaseLLM)
# ---------------------------------------------------------------------------

def _get_llm(model: str = "gpt-4o") -> BaseLLM:
    """Resolve a model string to a BaseLLM instance via LLMFactory."""
    if model.startswith("gemini"):
        return LLMFactory.create("gemini", model=model)
    return LLMFactory.create("openai", model=model)


def _pil_to_numpy(img: "Image.Image") -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def query_vlm(
    grid_image: Union["Image.Image", np.ndarray],
    prompt: str,
    model: str = "gpt-4o",
    llm: Optional[BaseLLM] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Send a grid image + prompt to a VLM and return the text response.

    Parameters
    ----------
    grid_image : PIL Image or numpy array
    prompt : text prompt (user role)
    model : model name (used only if *llm* is None)
    llm : pre-constructed BaseLLM instance (takes priority over *model*)
    system_prompt : if provided, sent as a separate system-role message
    """
    if llm is None:
        llm = _get_llm(model)

    if hasattr(grid_image, "save"):
        arr = _pil_to_numpy(grid_image)
    else:
        arr = np.asarray(grid_image, dtype=np.uint8)

    if system_prompt is not None:
        return llm.make_multimodal_request(system_prompt, prompt, arr, temperature=0.0)
    return llm.make_text_and_image_request(prompt, arr, temperature=0.0)
