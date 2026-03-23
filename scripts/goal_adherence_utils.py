"""
Shared utilities for goal adherence: frame sampling, grid building, VLM queries,
and diary-based completion checking.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

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

DRONE_GOAL_MONITOR_CONTEXT = (
    "You are a goal adherence monitor for a drone. The frames are from the drone's first-person view. "
)

def build_prompt_what_changed(subtask: str) -> str:
    """Build the 'what changed' prompt: concise, one sentence, key facts only."""
    return f"""{DRONE_GOAL_MONITOR_CONTEXT}The relevant subtask is: {subtask}

These two images are consecutive moments in time (first then second). In **one short sentence**, state what changed between the first and second image relative to this subtask. Be concise: no extraneous detail, no repetition of the subtask. Your description must be **strictly relevant** to the subtask (e.g. for \"move past the traffic light\", the light changing color is not relevant—only whether the drone's position relative to the light changed). Include only key facts that directly bear on the subtask: object appeared or disappeared, got bigger or smaller, or the drone passed it. Examples: object of interest got closer/bigger; object no longer visible or much smaller; object came into view."""

PROMPT_SUBTASK_COMPLETE = (
    "Given the diary of what changed between each pair of moments and the grid of frames so far, "
    "has the drone completed this subtask? Answer with exactly: Yes the subtask is complete. "
    "OR: No the subtask is not complete."
)


def sample_frames_every_n(frames: Sequence[T], n: int) -> List[T]:
    """
    From an ordered sequence of frames (earliest to latest), return the last frame
    and every n-th frame backward to the start, in strict temporal order.

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
    """
    Discover frame paths in a results directory (LTL or REPL layout).
    Returns paths in temporal order.

    - LTL: dir_path/frames/*.png
    - REPL: dir_path/frame_*.png
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
    padding_px: int = 16,
    max_size: int = 2048,
    min_short_side: int = 768,
) -> "Image.Image":
    """
    Build a single image grid from ordered frame paths with white padding between cells.

    Formula: width = cols * frame_w + (cols + 1) * padding_px,
             height = rows * frame_h + (rows + 1) * padding_px.
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
    if cell_w < 1:
        cell_w = 1
    if cell_h < 1:
        cell_h = 1

    W = cols * cell_w + (cols + 1) * p
    H = rows * cell_h + (rows + 1) * p
    W, H = int(W), int(H)

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


def query_vlm(
    grid_image: Union["Image.Image", "np.ndarray"],
    prompt: str,
    model: str = "gpt-4o",
    api_key: Union[str, None] = None,
) -> str:
    """
    Send the grid image to a VLM with detail="high" and return the model's text response.
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
            "OPENAI_API_KEY must be set or pass api_key to query_vlm"
        )

    if hasattr(grid_image, "save"):
        buf = io.BytesIO()
        grid_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    else:
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
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    **grid_kwargs,
) -> str:
    """
    Load frames (from dir or list), sample every n, build grid, query the model for temporal description.
    """
    if isinstance(frame_paths, (Path, str)):
        frame_paths = get_ordered_frames_from_dir(Path(frame_paths))
    if not frame_paths:
        raise ValueError("No frames found (empty directory or list)")

    sampled = sample_frames_every_n(frame_paths, n)
    if not sampled:
        raise ValueError("No frames after sampling")

    grid = build_frame_grid(sampled, **grid_kwargs)
    return query_vlm(
        grid, prompt or DEFAULT_TEMPORAL_PROMPT, model=model, api_key=api_key
    )


query_gpt4o_mini_temporal = query_vlm


def parse_yes_no_response(text: str) -> Optional[bool]:
    """
    Parse a response for Yes the subtask is complete vs No the subtask is not complete.
    Returns True if complete, False if not complete, None if ambiguous.
    """
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
    grid_image: Union["Image.Image", Any],
    subtask: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
) -> str:
    """
    Ask the model what changed between the first and second image in a 2-frame grid,
    relative to the given subtask.
    """
    prompt = build_prompt_what_changed(subtask)
    return query_vlm(grid_image, prompt, model=model, api_key=api_key)


def check_subtask_complete_diary(
    frame_paths: Union[Path, str, List[Path]],
    subtask: str,
    n: int,
    api_key: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
    model: str = "gpt-4o",
    **grid_kwargs: Any,
) -> dict:
    """
    Run the diary-based completion check: at each checkpoint, query what changed
    (append to diary) and whether the subtask is complete; return when complete or end.

    Returns:
        dict with keys: complete (bool), frame_index (int | None), diary (list[str]),
        reasoning (str | None), last_checkpoint (int | None).
    """
    def _frame_source() -> str:
        if isinstance(frame_paths, (Path, str)):
            return str(Path(frame_paths))
        return f"list of {len(frame_paths)} paths"

    if isinstance(frame_paths, (Path, str)):
        frames = get_ordered_frames_from_dir(Path(frame_paths))
    else:
        frames = list(frame_paths)

    result_empty = {
        "complete": False,
        "frame_index": None,
        "diary": [],
        "reasoning": None,
        "last_checkpoint": None,
    }
    if not frames:
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            config = {"subtask": subtask, "n": n, "model": model, "frame_source": _frame_source(), "total_frames": 0, "grid_kwargs": grid_kwargs}
            (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))
            (artifacts_dir / "summary.json").write_text(json.dumps(result_empty, indent=2))
        return result_empty

    T = len(frames)
    if T < n + 1:
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            config = {"subtask": subtask, "n": n, "model": model, "frame_source": _frame_source(), "total_frames": T, "grid_kwargs": grid_kwargs}
            (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))
            (artifacts_dir / "summary.json").write_text(json.dumps(result_empty, indent=2))
        return result_empty

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config = {"subtask": subtask, "n": n, "model": model, "frame_source": _frame_source(), "total_frames": T, "grid_kwargs": grid_kwargs}
        (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))

    diary: List[str] = []
    last_checkpoint: Optional[int] = None

    k = 1
    while k * n < T:
        current = k * n
        last_checkpoint = current

        two_paths = [frames[current - n], frames[current]]
        grid_two = build_frame_grid(two_paths, **grid_kwargs)
        prompt_what_changed = build_prompt_what_changed(subtask)
        change_text = query_what_changed_between_frames(
            grid_two, subtask, api_key=api_key, model=model
        )
        diary.append(
            f"Between frame {current - n} and frame {current}: {change_text}"
        )

        if artifacts_dir is not None:
            cp_dir = artifacts_dir / f"checkpoint_{current}"
            cp_dir.mkdir(parents=True, exist_ok=True)
            grid_two.save(cp_dir / "grid_two_frames.png")
            (cp_dir / "prompt_what_changed.txt").write_text(prompt_what_changed)
            (cp_dir / "response_what_changed.txt").write_text(change_text)

        capped = frames[: current + 1]
        sampled = sample_frames_every_n(capped, n)
        if not sampled:
            k += 1
            continue
        grid_full = build_frame_grid(sampled, **grid_kwargs)
        diary_blob = "\n".join(diary)
        prompt_b = (
            f"{DRONE_GOAL_MONITOR_CONTEXT}Subtask: {subtask}\n\n"
            f"Diary of what changed between consecutive moments:\n{diary_blob}\n\n"
            f"{PROMPT_SUBTASK_COMPLETE}"
        )
        response_b = query_vlm(
            grid_full, prompt_b, model=model, api_key=api_key
        )

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
