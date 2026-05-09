"""Compose final replay_synced.mp4 via ffmpeg.

Build steps:
  1. onboard.mp4: frames concatenated with per-frame durations from
     recording_log.jsonl (frame N held until frame N+1's timestamp).
  2. panels.mp4: panel PNGs concatenated at 1 second each, then encoded
     at 30 fps.
  3. final composite: panels.mp4 base + phone clip (top-right, fades in
     at phone_offset_s) + onboard.mp4 (bottom-right, fades in at
     onboard_first_t). Phone is padded by 2 s with tpad so it does not
     run out before on-board ends.

Run as a CLI:
    python -m scripts.replay_video.composite <results_dir> [<out_path>]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

from . import render as render_mod
from . import timeline as tl_mod


def _ffmpeg() -> str:
    # imageio-ffmpeg ships an ffmpeg binary; prefer it if available.
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _resolve_frame_path(frame_path: str, results_dir: Path) -> str:
    """Return a valid local path for a frame.

    The recording_log.jsonl may contain absolute paths from the machine where
    the run was recorded (e.g. /home/choukram/repos/...). If the stored path
    does not exist, fall back to results_dir/frames/<filename>.
    """
    p = Path(frame_path)
    if p.exists():
        return str(p)
    # Derive filename and look it up locally.
    local = results_dir / "frames" / p.name
    if local.exists():
        return str(local)
    raise FileNotFoundError(
        f"Cannot find frame {p.name}: tried {frame_path} and {local}"
    )


def _write_concat_demuxer_file(path: Path, entries: List[tuple]) -> None:
    """Write an ffmpeg concat demuxer file. entries = [(file_path, duration_s)]."""
    with path.open("w") as f:
        f.write("ffconcat version 1.0\n")
        for fp, dur in entries:
            f.write(f"file '{fp}'\n")
            f.write(f"duration {dur:.6f}\n")
        # Repeat the last file once with no duration so ffmpeg encodes the final frame.
        if entries:
            last_fp = entries[-1][0]
            f.write(f"file '{last_fp}'\n")


def build_onboard_mp4(tl, out_path: Path, ffmpeg: str, results_dir: Path) -> None:
    list_path = out_path.with_suffix(".txt")
    entries = []
    for i, fr in enumerate(tl.frames):
        resolved = _resolve_frame_path(fr.frame_path, results_dir)
        if i + 1 < len(tl.frames):
            dur = max(0.001, tl.frames[i + 1].t - fr.t)
        else:
            dur = 0.1
        entries.append((resolved, dur))
    _write_concat_demuxer_file(list_path, entries)

    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vsync", "cfr", "-r", "30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def build_panels_mp4(panel_pngs: Dict[int, Path], duration_s: float, out_path: Path, ffmpeg: str) -> None:
    """Each panel held 1 second; ffmpeg upsamples to 30 fps."""
    list_path = out_path.with_suffix(".txt")
    entries = []
    seconds = sorted(panel_pngs.keys())
    for s in seconds:
        entries.append((str(panel_pngs[s]), 1.0))
    _write_concat_demuxer_file(list_path, entries)

    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vsync", "cfr", "-r", "30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-t", f"{duration_s:.3f}",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def composite_final(
    panels_mp4: Path,
    phone_mp4: Path,
    onboard_mp4: Path,
    out_path: Path,
    canvas_w: int,
    canvas_h: int,
    right_col_w: int,
    phone_h: int,
    onboard_h: int,
    pad_l: int,
    pad_t_phone: int,
    pad_t_onboard: int,
    duration_s: float,
    phone_offset_s: float,
    onboard_first_t: float,
    ffmpeg: str,
) -> None:
    # The right column starts at x = canvas_w - pad_l - right_col_w (we pad
    # both sides equally with the canvas padding).
    x_right = canvas_w - 8 - right_col_w
    y_phone = pad_t_phone
    y_onboard = pad_t_onboard

    # Phone needs to freeze on its last frame for ~1.4s after it ends.
    # tpad stop_mode=clone with stop_duration=2 is generous and harmless.
    filter_complex = (
        f"[1:v]scale={right_col_w}:{phone_h},setsar=1,tpad=stop_mode=clone:stop_duration=2[ph];"
        f"[2:v]scale={right_col_w}:{onboard_h},setsar=1[ob];"
        f"[0:v][ph]overlay={x_right}:{y_phone}:enable='gte(t,{phone_offset_s:.3f})'[v1];"
        f"[v1][ob]overlay={x_right}:{y_onboard}:enable='gte(t,{onboard_first_t:.3f})'[vout]"
    )

    cmd = [
        ffmpeg, "-y",
        "-i", str(panels_mp4),
        "-i", str(phone_mp4),
        "-i", str(onboard_mp4),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def build_2x_from_1x(input_mp4: Path, out_path: Path, ffmpeg: str) -> None:
    """Produce a 2x-speed copy of input_mp4 at out_path using ffmpeg setpts.

    Halves presentation timestamps and re-encodes at 30 fps. No audio.
    """
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_mp4),
        "-vf", "setpts=PTS/2",
        "-r", "30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def build(results_dir: Path, out_path: Path) -> None:
    results_dir = results_dir.resolve()
    tmp_dir = results_dir / "_replay_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tl = tl_mod.load(results_dir)
    ffmpeg = _ffmpeg()

    # 1. Render panels (idempotent: skip if already rendered with matching second count).
    panels_dir = tmp_dir / "panels"
    panel_pngs = render_mod.render_all(results_dir, panels_dir)

    # 2. Build onboard.mp4
    onboard_mp4 = tmp_dir / "onboard.mp4"
    build_onboard_mp4(tl, onboard_mp4, ffmpeg, results_dir)

    # 3. Build panels.mp4
    panels_mp4 = tmp_dir / "panels.mp4"
    build_panels_mp4(panel_pngs, tl.duration_s, panels_mp4, ffmpeg)

    # 4. Find phone clip
    phone_files = sorted(results_dir.glob("[0-9]" * 8 + "_" + "[0-9]" * 6 + ".mp4"))
    phone_mp4 = phone_files[0]

    # 5. Layout numbers (must match panel.html / spec)
    canvas_w, canvas_h = 1920, 1080
    pad = 8
    gap = 6
    topbar_h = 30
    inner_w = canvas_w - 2 * pad
    right_col_w = int(round(0.34 * inner_w))             # 34% of inner width
    # Right column is a 9fr / 16fr CSS grid that splits the COLUMN HEIGHT,
    # not width. Column height = canvas - top/bottom pad - topbar - 1 gap (between
    # topbar and the right column itself). One additional gap sits between the
    # phone and on-board cells, so the two cell heights sum to col_h - gap.
    col_h = canvas_h - 2 * pad - topbar_h - gap
    phone_h = int(round((col_h - gap) * 9 / 25))
    onboard_h = int(round((col_h - gap) * 16 / 25))
    pad_t_phone = pad + topbar_h + gap
    pad_t_onboard = pad_t_phone + phone_h + gap

    composite_final(
        panels_mp4=panels_mp4,
        phone_mp4=phone_mp4,
        onboard_mp4=onboard_mp4,
        out_path=out_path,
        canvas_w=canvas_w, canvas_h=canvas_h,
        right_col_w=right_col_w,
        phone_h=phone_h, onboard_h=onboard_h,
        pad_l=pad, pad_t_phone=pad_t_phone, pad_t_onboard=pad_t_onboard,
        duration_s=tl.duration_s,
        phone_offset_s=tl.phone_offset_s,
        onboard_first_t=tl.onboard_first_t,
        ffmpeg=ffmpeg,
    )

    # 6. Derive a 2x-speed copy from the 1x output for compact viewing.
    out_2x_path = out_path.with_name(out_path.stem + "_2x" + out_path.suffix)
    build_2x_from_1x(out_path, out_2x_path, ffmpeg)


def main():
    parser = argparse.ArgumentParser(description="Compose final replay video")
    parser.add_argument("results_dir")
    parser.add_argument("out_path", nargs="?", default=None)
    args = parser.parse_args()
    out_path = Path(args.out_path) if args.out_path else Path(args.results_dir).resolve() / "replay_synced.mp4"
    build(Path(args.results_dir), out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
