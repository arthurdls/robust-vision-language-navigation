"""Render the panel HTML to PNG snapshots, one per output second.

We render a panel for each integer second in [0, ceil(duration_s)). The
panel HTML carries placeholder boxes where the videos go; ffmpeg overlays
the real video pixels later. Rendering is cached by a content-hash so
duplicate states reuse the same PNG file.

Run as a CLI:
    python -m scripts.replay_video.render <results_dir> <out_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict

from playwright.sync_api import sync_playwright

from . import timeline as tl_mod

# The timer's bar_time changes every second, so it's part of the hash if we
# want a different file per second. Set to False to share PNGs across seconds
# when content (other than the timer) is identical; True (default) renders
# one PNG per second so the topbar timer is always accurate.
INCLUDE_TIMER_IN_HASH = True


def _state_hash(state: dict) -> str:
    if INCLUDE_TIMER_IN_HASH:
        payload = state
    else:
        payload = {k: v for k, v in state.items() if k != "bar_time"}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def render_all(results_dir: Path, out_dir: Path) -> Dict[int, Path]:
    """Render one PNG per integer output second. Returns {second: png_path}."""
    # Resolve to absolute so that image paths inside timeline are absolute,
    # which lets panel.js construct valid file:// URLs.
    results_dir = results_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tl = tl_mod.load(results_dir)
    template_path = Path(__file__).parent / "templates" / "panel.html"

    seconds = list(range(math.ceil(tl.duration_s) + 1))
    print(f"render: {len(seconds)} seconds, duration={tl.duration_s:.2f}")

    cache: Dict[str, Path] = {}
    output: Dict[int, Path] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--allow-file-access-from-files"],
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.goto(template_path.as_uri())
        page.wait_for_load_state("domcontentloaded")
        # panel.js loads via <script src="panel.js"> relative to the HTML.

        for second in seconds:
            state = tl_mod.state_at(tl, float(second))
            h = _state_hash(state)
            if h in cache:
                output[second] = cache[h]
                continue
            page.evaluate("(state) => window.hydrate(state)", state)
            page.wait_for_function(
                "Array.from(document.images).every(img => img.complete || img.src === '' || img.src === 'about:blank')",
                timeout=5000,
            )
            png_path = out_dir / f"panel_{second:04d}_{h}.png"
            page.screenshot(
                path=str(png_path),
                full_page=False,
                omit_background=False,
                clip={"x": 0, "y": 0, "width": 1920, "height": 1080},
            )
            cache[h] = png_path
            output[second] = png_path
            if second % 30 == 0:
                print(f"  rendered second {second}/{len(seconds) - 1}")

        browser.close()

    print(f"render: {len(output)} seconds rendered, {len(cache)} unique PNGs cached")
    return output


def main():
    parser = argparse.ArgumentParser(description="Render replay panels to PNG")
    parser.add_argument("results_dir")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    render_all(Path(args.results_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
