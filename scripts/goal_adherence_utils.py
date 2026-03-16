"""
Shared utilities for goal adherence approaches: frame sampling and discovery.
"""

import math
import os
from pathlib import Path
from typing import List, Sequence, Union

try:
    from PIL import Image
except ImportError:
    Image = None

from typing import TypeVar

T = TypeVar("T")

DEFAULT_TEMPORAL_PROMPT = (
    "These images are sequential frames from a single video, in temporal order "
    "(left to right, top to bottom). Describe what is happening across the frames."
)


def sample_frames_every_n(frames: Sequence[T], n: int) -> List[T]:
    """
    From an ordered sequence of frames (earliest to latest), return the last frame
    and every n-th frame backward to the start, in strict temporal order.

    Example: 144 frames, n=10 -> indices [3, 13, 23, ..., 133, 143].

    Args:
        frames: Ordered sequence of frames (paths, arrays, etc.).
        n: Step size; must be >= 1.

    Returns:
        Sampled frames in strict temporal order (earliest first).
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
    """
    Discover frame paths in a results directory (LTL or REPL layout).
    Returns paths in temporal order.

    - LTL: dir_path/frames/*.png
    - REPL: dir_path/frame_*.png

    Args:
        dir_path: Path to the run/results directory.

    Returns:
        Sorted list of paths to frame images.
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


def build_frame_grid(
    frame_paths: List[Path],
    padding_px: int = 4,
    max_size: int = 2048,
    min_short_side: int = 768,
) -> "Image.Image":
    """
    Build a single image grid from ordered frame paths with white padding between cells.
    Grid fits within max_size x max_size with shortest side >= min_short_side.

    Args:
        frame_paths: Ordered list of paths to frame images.
        padding_px: White space between cells and margin around grid (default 4).
        max_size: Maximum width and height of the grid.
        min_short_side: Minimum length of the grid's shortest side.

    Returns:
        PIL Image of the grid (RGB).
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
    margin = p

    # Use first frame aspect ratio for all cells
    w0, h0 = frames[0].size
    aspect = w0 / h0 if h0 else 1.0

    # Total size: W = cols*cell_w + (cols-1)*p + 2*margin, H = rows*cell_h + (rows-1)*p + 2*margin
    # cell_w/cell_h = aspect => cell_w = aspect * cell_h
    # So W = cols*aspect*cell_h + (cols-1)*p + 2*margin, H = rows*cell_h + (rows-1)*p + 2*margin
    # Constrain: W <= max_size, H <= max_size, min(W,H) >= min_short_side
    extra_w = (cols - 1) * p + 2 * margin
    extra_h = (rows - 1) * p + 2 * margin

    # Max cell_h from W and H limits
    cell_h_from_w = (max_size - extra_w) / (cols * aspect) if cols * aspect else 0
    cell_h_from_h = (max_size - extra_h) / rows if rows else 0
    cell_h_max = min(cell_h_from_w, cell_h_from_h)

    # Min cell_h so that min(W,H) >= min_short_side
    cell_h_min_w = (min_short_side - extra_w) / (cols * aspect) if cols * aspect else 0
    cell_h_min_h = (min_short_side - extra_h) / rows if rows else 0
    cell_h_min = max(cell_h_min_w, cell_h_min_h, 1)

    # Use largest feasible cell size
    cell_h = cell_h_max if cell_h_max >= max(cell_h_min, 1) else max(cell_h_min, 1)
    cell_w = aspect * cell_h
    cell_w, cell_h = int(cell_w), int(cell_h)
    if cell_w < 1:
        cell_w = 1
    if cell_h < 1:
        cell_h = 1

    W = cols * cell_w + (cols - 1) * p + 2 * margin
    H = rows * cell_h + (rows - 1) * p + 2 * margin
    W, H = int(W), int(H)

    grid = Image.new("RGB", (W, H), (255, 255, 255))

    for idx, im in enumerate(frames):
        r = idx // cols
        c = idx % cols
        # Resize frame to fit inside cell preserving aspect ratio
        im_w, im_h = im.size
        scale = min(cell_w / im_w, cell_h / im_h, 1.0) if im_w and im_h else 1.0
        new_w = max(1, int(im_w * scale))
        new_h = max(1, int(im_h * scale))
        im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x = margin + c * (cell_w + p) + (cell_w - new_w) // 2
        y = margin + r * (cell_h + p) + (cell_h - new_h) // 2
        grid.paste(im_resized, (x, y))

    return grid


def query_gpt4o_mini_temporal(
    grid_image: Union["Image.Image", "np.ndarray"],
    prompt: str,
    model: str = "gpt-4o-mini",
    api_key: Union[str, None] = None,
) -> str:
    """
    Send the grid image to gpt-4o-mini with detail="high" and return the model's text response.

    Args:
        grid_image: PIL Image or numpy array (H,W,3 RGB).
        prompt: Text prompt asking what is happening across the frames.
        model: Model name (default gpt-4o-mini).
        api_key: OpenAI API key; if None, uses OPENAI_API_KEY env.

    Returns:
        Assistant message content string.
    """
    import base64
    import io

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai is required. Install with: pip install openai"
        ) from e

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY must be set or pass api_key to query_gpt4o_mini_temporal"
        )

    if hasattr(grid_image, "save"):
        # PIL Image
        buf = io.BytesIO()
        grid_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    else:
        # Assume numpy array
        if Image is None:
            raise RuntimeError("PIL required to encode numpy image")
        pil = Image.fromarray(grid_image.astype("uint8"))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    data_uri = f"data:image/png;base64,{b64}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "high"},
                },
            ],
        }
    ]
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    content = resp.choices[0].message.content
    return content or ""


def analyze_temporal_frames(
    frame_paths: Union[Path, str, List[Path]],
    n: int,
    prompt: Union[str, None] = None,
    **grid_kwargs,
) -> str:
    """
    Load frames (from dir or list), sample every n, build grid, query gpt-4o-mini for temporal description.

    Args:
        frame_paths: Directory path (Path or str) or ordered list of frame paths.
        n: Sampling step (sample_frames_every_n).
        prompt: Optional prompt; default describes temporal order and asks what is happening.
        **grid_kwargs: Passed to build_frame_grid (e.g. padding_px, max_size, min_short_side).

    Returns:
        Model response text.
    """
    if isinstance(frame_paths, (Path, str)):
        frame_paths = get_ordered_frames_from_dir(Path(frame_paths))
    if not frame_paths:
        raise ValueError("No frames found (empty directory or list)")

    sampled = sample_frames_every_n(frame_paths, n)
    if not sampled:
        raise ValueError("No frames after sampling")

    grid = build_frame_grid(sampled, **grid_kwargs)
    return query_gpt4o_mini_temporal(grid, prompt or DEFAULT_TEMPORAL_PROMPT)
