"""
Shared utilities for goal adherence approaches: frame sampling and discovery.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

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

# Context added to prompts for both approaches: goal adherence for a drone.
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

# Few-shot approach 2: pre-computed reasoning for examples, CoT only for final grid
# Constraint: reason only about what directly bears on the subtask; ignore irrelevant details.
FEWSHOT_SUBTASK_FOCUS = (
    "Focus strictly on what the subtask asks. Do not mention details that are irrelevant to the subtask "
    "(e.g. for \"move past the traffic light\", the light's color is irrelevant—only whether the drone has moved past it). "
    "Mention only: position or movement relative to the relevant object, or whether the goal is achieved."
)

FEWSHOT_COT_INSTRUCTION = (
    DRONE_GOAL_MONITOR_CONTEXT
    + "You will see several example grids, each with pre-written reasoning and a label (Complete or Not complete). "
    "Then you will see one final grid. "
    "For the final grid only, reason step-by-step across the frames (left to right, earliest to latest) relative to the subtask. "
    + FEWSHOT_SUBTASK_FOCUS + " "
    "Then on the very last line of your response, give only the final answer: **Yes the subtask is complete.** OR **No the subtask is not complete.**"
)

FEWSHOT_FINAL_QUESTION_TEMPLATE = (
    "**Final grid.** Subtask: {subtask}. "
    "Reason step-by-step across these frames (left to right, earliest to latest) relative to the subtask. "
    + FEWSHOT_SUBTASK_FOCUS + " "
    "Then on the **very last line** of your response, give only: **Yes the subtask is complete.** OR **No the subtask is not complete.**"
)

# Prompt to pre-generate reasoning for one example grid (used in a separate API call per example)
FEWSHOT_EXAMPLE_REASONING_PROMPT = (
    "These frames are from a single drone run, in temporal order (left to right, top to bottom). "
    "Subtask for this run: {subtask}. "
    "This run is labeled {label}. "
    "In 2–4 sentences, reason across the frames and explain why it is labeled {label}. "
    "Reason only about what is strictly relevant to the subtask. Do not mention irrelevant details "
    "(e.g. for \"move past the traffic light\", do not mention the light's color—only whether the drone has passed it). "
    "Output only the reasoning; no prefix or final answer."
)


def get_example_reasoning(
    grid_image: Union["Image.Image", Any],
    subtask: str,
    label: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Pre-compute reasoning for one example grid. One API call per example.
    label is 'complete' or 'not_complete'; subtask is the example's subtask.
    """
    label_display = "Complete" if label == "complete" else "Not complete"
    prompt = FEWSHOT_EXAMPLE_REASONING_PROMPT.format(
        subtask=subtask or "the given goal",
        label=label_display,
    )
    return query_gpt4o_mini_temporal(grid_image, prompt, model=model, api_key=api_key).strip()


def build_fewshot_example_with_reasoning(
    example_index: int,
    example_subtask: Optional[str],
    label: str,
    reasoning: str,
) -> str:
    """Build per-example text with pre-computed reasoning (for prompt of interest)."""
    parts = [f"Example {example_index}:"]
    if example_subtask:
        parts.append(f" Subtask for this example: {example_subtask}.")
    parts.append(f" Label: {label.replace('_', ' ').title()}.")
    parts.append(f" Reasoning: {reasoning}")
    return "".join(parts)


def _find_final_answer_line(text: str) -> Optional[Tuple[int, int, bool]]:
    """
    Find the line in text that contains the final Yes/No answer.
    Returns (start, end, is_yes) or None. Prefers last non-empty line, then line with Answer:/FINAL ANSWER:.
    """
    if not text or not text.strip():
        return None
    lines = text.splitlines()
    # Last non-empty line and its span in text
    last_line_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_line_idx = i
            break
    if last_line_idx is None:
        return None
    # Start position of line at last_line_idx
    pos = 0
    for i in range(last_line_idx):
        pos = text.find("\n", pos)
        if pos == -1:
            break
        pos += 1
    start = pos
    end = text.find("\n", start)
    if end == -1:
        end = len(text)
    last_content = text[start:end].strip().lower()
    if "yes the subtask is complete" in last_content or "yes, the subtask is complete" in last_content:
        return (start, end, True)
    if "no the subtask is not complete" in last_content or "no, the subtask is not complete" in last_content:
        return (start, end, False)
    # Optional: line starting with Answer: or FINAL ANSWER:
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip().lower()
        if ln.startswith("answer:") or ln.startswith("final answer:"):
            if "yes the subtask is complete" in ln or "yes, the subtask is complete" in ln:
                pos = 0
                for j in range(i):
                    pos = text.find("\n", pos) + 1
                e = text.find("\n", pos)
                if e == -1:
                    e = len(text)
                return (pos, e, True)
            if "no the subtask is not complete" in ln or "no, the subtask is not complete" in ln:
                pos = 0
                for j in range(i):
                    pos = text.find("\n", pos) + 1
                e = text.find("\n", pos)
                if e == -1:
                    e = len(text)
                return (pos, e, False)
    return None


def split_reasoning_and_answer(response: str) -> Tuple[str, str]:
    """
    Split response into reasoning (CoT) and final answer line.
    Uses same detection as parse_yes_no_response (last-line / marker preference).
    Returns (reasoning, final_answer_line); final_answer_line may be empty if not found.
    """
    if not response or not response.strip():
        return ("", "")
    found = _find_final_answer_line(response)
    if found is None:
        return (response.strip(), "")
    start, end, _ = found
    reasoning = response[:start].strip()
    final_line = response[start:end].strip()
    return (reasoning, final_line)


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
    padding_px: int = 16,
    max_size: int = 2048,
    min_short_side: int = 768,
) -> "Image.Image":
    """
    Build a single image grid from ordered frame paths with white padding between cells.
    Grid size is determined by each frame's resolution and padding (max_size and
    min_short_side are ignored for backward compatibility).

    Formula: width = cols * frame_w + (cols + 1) * padding_px,
             height = rows * frame_h + (rows + 1) * padding_px.

    Args:
        frame_paths: Ordered list of paths to frame images.
        padding_px: White space between cells and margin around grid (default 16).
        max_size: Ignored; kept for backward compatibility.
        min_short_side: Ignored; kept for backward compatibility.

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

    # Cell size = first frame size
    cell_w, cell_h = frames[0].size
    if cell_w < 1:
        cell_w = 1
    if cell_h < 1:
        cell_h = 1

    # Grid dimensions: (cols+1) and (rows+1) gaps of padding
    W = cols * cell_w + (cols + 1) * p
    H = rows * cell_h + (rows + 1) * p
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
        x = p + c * (cell_w + p) + (cell_w - new_w) // 2
        y = p + r * (cell_h + p) + (cell_h - new_h) // 2
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
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **grid_kwargs,
) -> str:
    """
    Load frames (from dir or list), sample every n, build grid, query the model for temporal description.

    Args:
        frame_paths: Directory path (Path or str) or ordered list of frame paths.
        n: Sampling step (sample_frames_every_n).
        prompt: Optional prompt; default describes temporal order and asks what is happening.
        model: OpenAI chat model (default gpt-4o-mini).
        api_key: OpenAI API key; if None, uses OPENAI_API_KEY env.
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
    return query_gpt4o_mini_temporal(
        grid, prompt or DEFAULT_TEMPORAL_PROMPT, model=model, api_key=api_key
    )


def parse_yes_no_response(text: str) -> Optional[bool]:
    """
    Parse a response for Yes the subtask is complete vs No the subtask is not complete.
    Prefers the last line (or line with Answer:/FINAL ANSWER:) then falls back to whole-text search.
    Returns True if complete, False if not complete, None if ambiguous.
    """
    if not text:
        return None
    # Prefer last-line / marker (same logic as split_reasoning_and_answer)
    found = _find_final_answer_line(text)
    if found is not None:
        _, _, is_yes = found
        return is_yes
    # Fall back to anywhere in text (backward compatibility)
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
    model: str = "gpt-4o-mini",
) -> str:
    """
    Ask the model what changed between the first and second image in a 2-frame grid,
    relative to the given subtask. Uses a prompt that includes the subtask and examples.
    """
    prompt = build_prompt_what_changed(subtask)
    return query_gpt4o_mini_temporal(grid_image, prompt, model=model, api_key=api_key)


def _image_to_data_uri(grid_image: Union["Image.Image", Any]) -> str:
    """Encode PIL Image or numpy array to data:image/png;base64,..."""
    import base64
    import io

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
    return f"data:image/png;base64,{b64}"


def query_gpt4o_mini_multiple_images(
    content_parts: List[Union[Tuple[str, str], Tuple[str, Any]]],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> str:
    """
    Send one user message with multiple text and image parts (each image with detail=high).
    content_parts: list of ("text", "string") or ("image", pil_image) in order.
    Returns the assistant content string.
    """
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
            "OPENAI_API_KEY must be set or pass api_key"
        )

    content = []
    for kind, value in content_parts:
        if kind == "text":
            content.append({"type": "text", "text": value})
        elif kind == "image":
            data_uri = _image_to_data_uri(value)
            content.append({
                "type": "image_url",
                "image_url": {"url": data_uri, "detail": "high"},
            })
        else:
            raise ValueError(f"Unknown content part kind: {kind}")

    messages = [{"role": "user", "content": content}]
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def check_subtask_complete_diary(
    frame_paths: Union[Path, str, List[Path]],
    subtask: str,
    n: int,
    api_key: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
    model: str = "gpt-4o-mini",
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

        # Query A: what changed between (current - n) and current (relative to subtask)
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

        # Query B: full grid up to current + diary -> is subtask complete?
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
        response_b = query_gpt4o_mini_temporal(
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


def check_subtask_complete_fewshot(
    frame_paths: Union[Path, str, List[Path]],
    subtask: str,
    few_shot_examples: List[Tuple[Union[Path, str, List[Path]], Optional[int], Optional[str]]],
    n: int,
    api_key: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
    model: str = "gpt-4o-mini",
    **grid_kwargs: Any,
) -> Tuple[bool, str]:
    """
    Run the few-shot completion check: build example grids (with labels), current grid,
    one query; parse Yes/No. Returns (complete: bool, reasoning: str).
    few_shot_examples: list of (path_or_list, complete_at, example_subtask).
    example_subtask is the original task for that example (optional); if None, not shown.
    """
    current_run_str = str(Path(frame_paths)) if isinstance(frame_paths, (Path, str)) else f"list of {len(frame_paths)} paths"
    examples_config = []
    for tup in few_shot_examples:
        p, c = tup[0], tup[1]
        ex_sub = tup[2] if len(tup) > 2 else None
        examples_config.append({
            "path": str(Path(p)) if isinstance(p, (Path, str)) else f"list of {len(p)} paths",
            "complete_at": c,
            "subtask": ex_sub,
        })

    if isinstance(frame_paths, (Path, str)):
        current_frames = get_ordered_frames_from_dir(Path(frame_paths))
    else:
        current_frames = list(frame_paths)

    if not current_frames:
        fail_reason = "No frames in current run."
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            config = {"subtask": subtask, "n": n, "model": model, "current_run": current_run_str, "examples": examples_config, "grid_kwargs": grid_kwargs}
            (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))
            (artifacts_dir / "summary.json").write_text(json.dumps({"complete": False, "reasoning": fail_reason}, indent=2))
        return False, fail_reason

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config = {"subtask": subtask, "n": n, "model": model, "current_run": current_run_str, "examples": examples_config, "grid_kwargs": grid_kwargs}
        (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Pre-compute reasoning for each example (one API call per example), then build prompt with it
    example_parts: List[Union[Tuple[str, str], Tuple[str, Any]]] = []
    prompt_lines: List[str] = []
    example_parts.append(("text", FEWSHOT_COT_INSTRUCTION))
    prompt_lines.append(FEWSHOT_COT_INSTRUCTION)

    example_index = 0
    for i, tup in enumerate(few_shot_examples, 1):
        path_or_list = tup[0]
        complete_at = tup[1]
        example_subtask = tup[2] if len(tup) > 2 else None
        if isinstance(path_or_list, (Path, str)):
            frames = get_ordered_frames_from_dir(Path(path_or_list))
        else:
            frames = list(path_or_list)
        if not frames:
            continue
        if complete_at is not None:
            capped = frames[: complete_at + 1]
        else:
            capped = frames
        sampled = sample_frames_every_n(capped, n)
        if not sampled:
            continue
        grid = build_frame_grid(sampled, **grid_kwargs)
        label = "complete" if complete_at is not None else "not_complete"
        reasoning = get_example_reasoning(
            grid, example_subtask or subtask, label, api_key=api_key, model=model
        )
        example_index += 1
        if artifacts_dir is not None:
            grid.save(artifacts_dir / f"example_{example_index:02d}_label_{label}.png")
            (artifacts_dir / f"example_{example_index:02d}_reasoning.txt").write_text(reasoning)
        example_label_text = build_fewshot_example_with_reasoning(
            i, example_subtask, label, reasoning
        )
        example_parts.append(("image", grid))
        example_parts.append(("text", example_label_text))
        prompt_lines.append(f"[Image: example {example_index}]")
        prompt_lines.append(example_label_text)

    # Current grid
    sampled_current = sample_frames_every_n(current_frames, n)
    if not sampled_current:
        fail_reason = "No frames after sampling."
        if artifacts_dir is not None:
            (artifacts_dir / "summary.json").write_text(json.dumps({"complete": False, "reasoning": fail_reason}, indent=2))
        return False, fail_reason

    grid_current = build_frame_grid(sampled_current, **grid_kwargs)
    final_text = FEWSHOT_FINAL_QUESTION_TEMPLATE.format(subtask=subtask)
    example_parts.append(("image", grid_current))
    example_parts.append(("text", final_text))

    if artifacts_dir is not None:
        grid_current.save(artifacts_dir / "current_grid.png")
        prompt_lines.append("[Image: current grid]")
        prompt_lines.append(final_text)
        (artifacts_dir / "prompt.txt").write_text("\n\n".join(prompt_lines))

    response = query_gpt4o_mini_multiple_images(
        example_parts, model=model, api_key=api_key
    )

    complete = parse_yes_no_response(response)
    if complete is None:
        complete = False

    if artifacts_dir is not None:
        (artifacts_dir / "response.txt").write_text(response)
        reasoning_text, final_answer_line = split_reasoning_and_answer(response)
        summary = {"complete": complete, "reasoning": reasoning_text}
        if final_answer_line:
            summary["final_answer"] = final_answer_line
        (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return complete, response


def check_subtask_complete_fewshot_checkpoints(
    frame_paths: Union[Path, str, List[Path]],
    subtask: str,
    few_shot_examples: List[Tuple[Union[Path, str, List[Path]], Optional[int], Optional[str]]],
    n: int,
    api_key: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
    model: str = "gpt-4o-mini",
    **grid_kwargs: Any,
) -> dict:
    """
    Run the few-shot completion check at checkpoints n, 2n, 3n, ...; save all
    intermediate inputs and outputs per checkpoint; stop when the model says complete.

    Returns dict with keys: complete (bool), frame_index (int | None), reasoning (str | None),
    last_checkpoint (int | None).
    """
    current_run_str = str(Path(frame_paths)) if isinstance(frame_paths, (Path, str)) else f"list of {len(frame_paths)} paths"
    examples_config = []
    for tup in few_shot_examples:
        p, c = tup[0], tup[1]
        ex_sub = tup[2] if len(tup) > 2 else None
        examples_config.append({
            "path": str(Path(p)) if isinstance(p, (Path, str)) else f"list of {len(p)} paths",
            "complete_at": c,
            "subtask": ex_sub,
        })

    if isinstance(frame_paths, (Path, str)):
        frames = get_ordered_frames_from_dir(Path(frame_paths))
    else:
        frames = list(frame_paths)

    result_incomplete = {
        "complete": False,
        "frame_index": None,
        "reasoning": None,
        "last_checkpoint": None,
    }
    if not frames:
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            config = {"subtask": subtask, "n": n, "model": model, "current_run": current_run_str, "examples": examples_config, "grid_kwargs": grid_kwargs}
            (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))
            (artifacts_dir / "summary.json").write_text(json.dumps(result_incomplete, indent=2))
        return result_incomplete

    T = len(frames)
    if T < n + 1:
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            config = {"subtask": subtask, "n": n, "model": model, "current_run": current_run_str, "examples": examples_config, "grid_kwargs": grid_kwargs}
            (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))
            (artifacts_dir / "summary.json").write_text(json.dumps(result_incomplete, indent=2))
        return result_incomplete

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config = {"subtask": subtask, "n": n, "model": model, "current_run": current_run_str, "examples": examples_config, "grid_kwargs": grid_kwargs}
        (artifacts_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Pre-compute reasoning for each example once (one API call per example), then build static parts
    static_parts: List[Union[Tuple[str, str], Tuple[str, Any]]] = []
    static_prompt_lines: List[str] = []
    static_parts.append(("text", FEWSHOT_COT_INSTRUCTION))
    static_prompt_lines.append(FEWSHOT_COT_INSTRUCTION)

    example_index = 0
    for i, tup in enumerate(few_shot_examples, 1):
        path_or_list = tup[0]
        complete_at = tup[1]
        example_subtask = tup[2] if len(tup) > 2 else None
        if isinstance(path_or_list, (Path, str)):
            example_frames = get_ordered_frames_from_dir(Path(path_or_list))
        else:
            example_frames = list(path_or_list)
        if not example_frames:
            continue
        if complete_at is not None:
            capped = example_frames[: complete_at + 1]
        else:
            capped = example_frames
        sampled = sample_frames_every_n(capped, n)
        if not sampled:
            continue
        grid = build_frame_grid(sampled, **grid_kwargs)
        label = "complete" if complete_at is not None else "not_complete"
        reasoning = get_example_reasoning(
            grid, example_subtask or subtask, label, api_key=api_key, model=model
        )
        example_index += 1
        if artifacts_dir is not None:
            grid.save(artifacts_dir / f"example_{example_index:02d}_label_{label}.png")
            (artifacts_dir / f"example_{example_index:02d}_reasoning.txt").write_text(reasoning)
        example_label_text = build_fewshot_example_with_reasoning(
            i, example_subtask, label, reasoning
        )
        static_parts.append(("image", grid))
        static_parts.append(("text", example_label_text))
        static_prompt_lines.append(f"[Image: example {example_index}]")
        static_prompt_lines.append(example_label_text)

    final_text_template = FEWSHOT_FINAL_QUESTION_TEMPLATE

    last_checkpoint: Optional[int] = None
    last_response: Optional[str] = None
    k = 1
    while k * n < T:
        current = k * n
        last_checkpoint = current

        current_frames = frames[: current + 1]
        sampled_current = sample_frames_every_n(current_frames, n)
        if not sampled_current:
            k += 1
            continue

        grid_current = build_frame_grid(sampled_current, **grid_kwargs)
        final_text = final_text_template.format(subtask=subtask)
        parts = list(static_parts) + [("image", grid_current), ("text", final_text)]

        cp_dir = None
        if artifacts_dir is not None:
            cp_dir = artifacts_dir / f"checkpoint_{current}"
            cp_dir.mkdir(parents=True, exist_ok=True)
            grid_current.save(cp_dir / "current_grid.png")
            prompt_lines = list(static_prompt_lines) + ["[Image: current grid]", final_text]
            (cp_dir / "prompt.txt").write_text("\n\n".join(prompt_lines))

        response = query_gpt4o_mini_multiple_images(
            parts, model=model, api_key=api_key
        )
        last_response = response

        if cp_dir is not None:
            (cp_dir / "response.txt").write_text(response)
            reasoning_text, final_answer_line = split_reasoning_and_answer(response)
            (cp_dir / "reasoning.txt").write_text(reasoning_text)

        complete = parse_yes_no_response(response)
        if complete is None:
            complete = False
        if complete:
            reasoning_text, _ = split_reasoning_and_answer(response)
            result = {
                "complete": True,
                "frame_index": current,
                "reasoning": reasoning_text,
                "last_checkpoint": current,
            }
            if artifacts_dir is not None:
                (artifacts_dir / "summary.json").write_text(json.dumps(result, indent=2))
            return result
        k += 1

    reasoning_text = ""
    if last_response:
        reasoning_text, _ = split_reasoning_and_answer(last_response)
    result = {
        "complete": False,
        "frame_index": None,
        "reasoning": reasoning_text or last_response,
        "last_checkpoint": last_checkpoint,
    }
    if artifacts_dir is not None:
        (artifacts_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result
