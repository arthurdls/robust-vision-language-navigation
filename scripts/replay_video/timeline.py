"""Parse hardware run artifacts into a sorted timeline of events.

The master clock t0 is run_info.json["start_time"]. All event `t` values
are seconds since t0 (float). Wall-clock timestamps come from:
  - recording_log.jsonl (per-frame ISO timestamp)
  - filesystem mtime on prompt/response/diary artifact files
  - filesystem mtime on convergence artifact files

Run as a CLI to dump the parsed event list:
    python -m scripts.replay_video.timeline /path/to/results/hardware
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


# Scale applied to x/y/z components when displaying poses in diary lines.
# The recording_log subgoal_rel_pose stores integrated dead-reckoning values
# that are roughly 40x larger than the actual physical movement (because the
# wire-level velocity scaling is not propagated into the stored pose). Scaling
# down by 0.025 makes the displayed distances match the drone's real motion
# at the clipped 0.2 m/s wire rate.
DIARY_TRANSLATION_SCALE = 0.025


@dataclass(frozen=True)
class FrameEvent:
    t: float
    step: int
    frame_path: str
    subgoal_index: int
    subgoal_rel_pose: tuple  # (x_m, y_m, z_m, yaw_deg) relative to subgoal start


@dataclass(frozen=True)
class CheckpointArtifact:
    t_local: float
    t_global: float
    t_diary: float
    subgoal_index: int      # 1-based, matches recording_log
    checkpoint_step: int    # parsed from directory name (checkpoint_NNNN)
    grid_local_path: Path
    grid_global_path: Path
    response_local_text: str
    response_global_json: dict
    diary_lines: List[str]


@dataclass(frozen=True)
class ConvergenceArtifact:
    t: float
    subgoal_index: int      # 1-based
    response_json: dict


@dataclass
class Timeline:
    t0: dt.datetime                     # absolute wall-clock for t=0
    duration_s: float                   # last frame timestamp minus t0
    phone_offset_s: float               # when phone clip's t=0 lands on master clock
    phone_duration_s: float             # phone clip length
    onboard_first_t: float              # first frame t (>= 0)
    onboard_last_t: float               # last frame t
    frames: List[FrameEvent]
    checkpoints: List[CheckpointArtifact]
    convergences: List[ConvergenceArtifact]
    subgoal_texts: List[str]            # 1-based via index 0 -> placeholder, 1..N real


def _parse_iso(s: str) -> dt.datetime:
    """Parse an ISO-8601 timestamp like 2026-05-07T13:53:22.621454."""
    # fromisoformat handles microseconds in py3.10+
    return dt.datetime.fromisoformat(s)


def _file_mtime(p: Path) -> dt.datetime:
    return dt.datetime.fromtimestamp(p.stat().st_mtime)


def _seconds_since(t0: dt.datetime, when: dt.datetime) -> float:
    return (when - t0).total_seconds()


def _load_frames(results_dir: Path, t0: dt.datetime) -> List[FrameEvent]:
    log_path = results_dir / "recording_log.jsonl"
    out: List[FrameEvent] = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sgrp = r.get("subgoal_rel_pose") or [0.0, 0.0, 0.0, 0.0]
            out.append(
                FrameEvent(
                    t=_seconds_since(t0, _parse_iso(r["timestamp"])),
                    step=int(r["step"]),
                    frame_path=r["frame_path"],
                    subgoal_index=int(r.get("subgoal_index", 1)),
                    subgoal_rel_pose=(float(sgrp[0]), float(sgrp[1]), float(sgrp[2]), float(sgrp[3])),
                )
            )
    out.sort(key=lambda e: e.t)
    return out


def _checkpoint_step_from_dir(name: str) -> int:
    # name is like "checkpoint_0028" or "checkpoint_0432"
    return int(name.split("_")[1])


def _load_checkpoints(subgoal_dirs, t0: dt.datetime) -> List[CheckpointArtifact]:
    out: List[CheckpointArtifact] = []
    for subgoal_index, sg_dir in subgoal_dirs:
        artifacts_dir = sg_dir / "diary_artifacts"
        if not artifacts_dir.is_dir():
            continue
        for cp_dir in sorted(artifacts_dir.iterdir()):
            if not cp_dir.name.startswith("checkpoint_"):
                continue
            local_path = cp_dir / "prompt_local.txt"
            global_path = cp_dir / "prompt_global.txt"
            diary_path = cp_dir / "diary.txt"
            resp_local_path = cp_dir / "response_local.txt"
            resp_global_path = cp_dir / "response_global.txt"
            grid_local_path = cp_dir / "grid_local.png"
            grid_global_path = cp_dir / "grid_global.png"

            # Skip entirely if we cannot establish a primary timestamp.
            if not local_path.exists():
                continue

            try:
                resp_global = json.loads(resp_global_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                resp_global = {}

            # t_global uses global prompt mtime when available, else local mtime + epsilon as
            # a fallback (ensures t_local < t_global even for incomplete checkpoints).
            if global_path.exists():
                t_global = _seconds_since(t0, _file_mtime(global_path))
            else:
                t_global = _seconds_since(t0, _file_mtime(local_path)) + 1e-3

            try:
                resp_local_text = resp_local_path.read_text().strip()
            except FileNotFoundError:
                resp_local_text = ""

            try:
                diary_lines = [ln for ln in diary_path.read_text().splitlines() if ln.strip()]
            except FileNotFoundError:
                diary_lines = []

            if diary_path.exists():
                t_diary = _seconds_since(t0, _file_mtime(diary_path))
            else:
                t_diary = _seconds_since(t0, _file_mtime(local_path))
            out.append(
                CheckpointArtifact(
                    t_local=_seconds_since(t0, _file_mtime(local_path)),
                    t_global=t_global,
                    t_diary=t_diary,
                    subgoal_index=subgoal_index,
                    checkpoint_step=_checkpoint_step_from_dir(cp_dir.name),
                    grid_local_path=grid_local_path,
                    grid_global_path=grid_global_path,
                    response_local_text=resp_local_text,
                    response_global_json=resp_global,
                    diary_lines=diary_lines,
                )
            )
    out.sort(key=lambda c: c.t_local)
    return out


def _load_convergences(subgoal_dirs, t0: dt.datetime) -> List[ConvergenceArtifact]:
    out: List[ConvergenceArtifact] = []
    for subgoal_index, sg_dir in subgoal_dirs:
        artifacts_dir = sg_dir / "diary_artifacts"
        if not artifacts_dir.is_dir():
            continue
        for cv_dir in sorted(artifacts_dir.iterdir()):
            if not cv_dir.name.startswith("convergence_"):
                continue
            resp_path = cv_dir / "response_00.txt"
            try:
                resp_json = json.loads(resp_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            out.append(
                ConvergenceArtifact(
                    t=_seconds_since(t0, _file_mtime(resp_path)),
                    subgoal_index=subgoal_index,
                    response_json=resp_json,
                )
            )
    out.sort(key=lambda c: c.t)
    return out


def _enumerate_subgoal_dirs(results_dir: Path):
    """Yield (1-based subgoal_index, subgoal directory path)."""
    pairs = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and d.name.startswith("subgoal_"):
            # Directory name is subgoal_NN_<slug>; index is 1-based.
            idx = int(d.name.split("_")[1])
            pairs.append((idx, d))
    pairs.sort(key=lambda p: p[0])
    return pairs


def load(results_dir: Path) -> Timeline:
    run_info = json.loads((results_dir / "run_info.json").read_text())
    t0 = _parse_iso(run_info["start_time"])

    frames = _load_frames(results_dir, t0)
    if not frames:
        raise RuntimeError(f"No frames found in {results_dir}/recording_log.jsonl")

    subgoal_dirs = _enumerate_subgoal_dirs(results_dir)
    checkpoints = _load_checkpoints(subgoal_dirs, t0)
    convergences = _load_convergences(subgoal_dirs, t0)

    subgoal_texts = ["(no subgoal)"]
    for s in run_info.get("subgoal_summaries", []):
        subgoal_texts.append(s.get("subgoal", ""))

    # Phone offset: use ffprobe to get exact creation_time (end) and duration.
    # Filename pattern: YYYYMMDD_HHMMSS.mp4
    phone_files = sorted(results_dir.glob("[0-9]" * 8 + "_" + "[0-9]" * 6 + ".mp4"))
    if not phone_files:
        raise RuntimeError(f"No phone .mp4 found in {results_dir} (pattern YYYYMMDD_HHMMSS.mp4)")
    phone_path = phone_files[0]
    # ffprobe gives us exact creation_time (the recording end) and duration.
    probe_result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(phone_path),
        ],
        capture_output=True, text=True,
    )
    if probe_result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (is it on PATH?): {probe_result.stderr.strip()}"
        )
    probe_data = json.loads(probe_result.stdout)
    fmt = probe_data["format"]
    phone_duration_s = float(fmt["duration"])
    # creation_time is UTC; convert to naive local via the run timezone offset.
    # The MP4 stores UTC; the run timestamps are local. We derive local creation
    # time by adding the UTC offset embedded in the MP4 (com.samsung.android.utc_offset).
    tags = fmt.get("tags", {})
    creation_time_utc_str = tags.get("creation_time", "")
    if creation_time_utc_str:
        # Strip trailing Z and parse as UTC, then shift to local using utc_offset tag.
        creation_time_utc_str = creation_time_utc_str.rstrip("Z").replace("T", " ")
        creation_time_utc = dt.datetime.fromisoformat(creation_time_utc_str.replace(" ", "T"))
        utc_offset_str = tags.get("com.samsung.android.utc_offset", "+0000")
        # utc_offset_str is like "-0400" or "+0000"
        sign = 1 if utc_offset_str[0] == "+" else -1
        h, m = int(utc_offset_str[1:3]), int(utc_offset_str[3:5])
        utc_offset = dt.timedelta(hours=h, minutes=m) * sign
        phone_end_local = creation_time_utc + utc_offset
    else:
        # Fallback: use mtime
        phone_end_local = dt.datetime.fromtimestamp(phone_path.stat().st_mtime)
    phone_start = phone_end_local - dt.timedelta(seconds=phone_duration_s)
    phone_offset_s = _seconds_since(t0, phone_start)

    return Timeline(
        t0=t0,
        duration_s=frames[-1].t,
        phone_offset_s=phone_offset_s,
        phone_duration_s=phone_duration_s,
        onboard_first_t=frames[0].t,
        onboard_last_t=frames[-1].t,
        frames=frames,
        checkpoints=checkpoints,
        convergences=convergences,
        subgoal_texts=subgoal_texts,
    )


def _format_t(t: float) -> str:
    minutes, seconds = divmod(t, 60)
    return f"{int(minutes):02d}:{seconds:06.3f}"


def main():
    parser = argparse.ArgumentParser(description="Dump replay-video timeline events")
    parser.add_argument("results_dir", help="path to results/hardware/")
    args = parser.parse_args()

    tl = load(Path(args.results_dir))
    print(f"t0:                {tl.t0.isoformat()}")
    print(f"duration_s:        {tl.duration_s:.3f}")
    print(f"phone_offset_s:    {tl.phone_offset_s:.3f}")
    print(f"phone_duration_s:  {tl.phone_duration_s:.3f}")
    print(f"onboard_first_t:   {tl.onboard_first_t:.3f}")
    print(f"onboard_last_t:    {tl.onboard_last_t:.3f}")
    print(f"frames:            {len(tl.frames)}")
    print(f"checkpoints:       {len(tl.checkpoints)}")
    print(f"convergences:      {len(tl.convergences)}")
    print()
    print("checkpoints:")
    for c in tl.checkpoints:
        gjson = c.response_global_json
        pct = gjson.get("completion_percentage")
        pct_str = f"{pct:.2f}" if pct is not None else "n/a"
        print(
            f"  t_local={_format_t(c.t_local)} t_global={_format_t(c.t_global)} "
            f"sg={c.subgoal_index} step={c.checkpoint_step} "
            f"completion={pct_str} "
            f"drive={gjson.get('drive_action')}"
        )
    print()
    print("convergences:")
    for c in tl.convergences:
        rj = c.response_json
        print(
            f"  t={_format_t(c.t)} sg={c.subgoal_index} "
            f"complete={rj.get('complete')} verdict={rj.get('diagnosis')}"
        )


def slug_for_subgoal_text(text: str, max_len: int = 12) -> str:
    """Derive a short snake_case slug from the converted instruction.

    Example: "Look to your left for ..." -> "look_left"
             "Move towards ..."          -> "move_towards"
    """
    import re
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return "step"
    if len(words) >= 2 and len(words[0]) + 1 + len(words[1]) <= max_len:
        return f"{words[0]}_{words[1]}"
    return words[0][:max_len]


def _format_bar_time(t: float, total: float) -> str:
    def fmt(x):
        m, s = divmod(max(0.0, x), 60)
        return f"{int(m):02d}:{int(s):02d}"
    return f"{fmt(t)} / {fmt(total)}"



def _pose_for_checkpoint(tl: "Timeline", subgoal_index: int, checkpoint_step: int) -> tuple:
    """Return the subgoal_rel_pose of the latest frame in the given subgoal at or before checkpoint_step.

    Scans all frames (no early break) because tl.frames is sorted by wall-clock t,
    not by (subgoal_index, step), and step resets at each subgoal boundary.
    """
    best = (0.0, 0.0, 0.0, 0.0)
    best_step = -1
    for fr in tl.frames:
        if fr.subgoal_index != subgoal_index:
            continue
        if fr.step <= checkpoint_step and fr.step > best_step:
            best = fr.subgoal_rel_pose
            best_step = fr.step
    return best


def _format_pose(pose: tuple) -> str:
    x, y, z, yaw = pose
    x *= DIARY_TRANSLATION_SCALE
    y *= DIARY_TRANSLATION_SCALE
    z *= DIARY_TRANSLATION_SCALE
    return f"[x: {x:.2f} m, y: {y:.2f} m, z: {z:.2f} m, yaw: {yaw:.1f}°]"


def _global_chips(resp: dict) -> list:
    chips = [
        {"label": f"complete: {str(resp.get('complete')).lower()}", "kind": ""},
        {"label": f"completion: {resp.get('completion_percentage', 0):.2f}", "kind": "ok"},
        {"label": f"should_stop: {str(resp.get('should_stop')).lower()}", "kind": "bad" if resp.get('should_stop') else ""},
        {"label": f"drive: {resp.get('drive_action', 'stop')}", "kind": "warn"},
    ]
    return chips


def _conv_chips(resp: dict, subgoal_index: int) -> list:
    diag = resp.get("diagnosis") or ("complete" if resp.get("complete") else "stopped_short")
    corrective = resp.get("corrective_instruction") or "null"
    return [
        {"label": f"verdict: {diag}", "kind": "ok" if diag == "complete" else "warn"},
        {"label": f"corrective: {corrective}", "kind": "" if corrective == "null" else "warn"},
        {"label": f"completion: {resp.get('completion_percentage', 0):.2f}", "kind": "ok"},
        {"label": f"subgoal: {subgoal_index}", "kind": ""},
    ]


def state_at(tl: Timeline, t: float) -> dict:
    """Return the dict to feed window.hydrate() at time t (seconds since t0)."""
    # Active subgoal: latest frame at or before t.
    active_sg = 1
    for fr in tl.frames:
        if fr.t > t:
            break
        active_sg = fr.subgoal_index

    sg_text = tl.subgoal_texts[active_sg] if 0 <= active_sg < len(tl.subgoal_texts) else ""
    sg_slug = slug_for_subgoal_text(sg_text)

    # Latest local checkpoint where t_local <= t.
    # tl.checkpoints is sorted by t_local, so a simple ordered scan works.
    latest_local = None
    for c in tl.checkpoints:
        if c.t_local <= t:
            latest_local = c
        else:
            break

    # Latest global checkpoint where t_global <= t AND a real response exists.
    # Synthetic checkpoints (in the interrupted subgoal 4) have empty
    # response_global_json={}; skip those.
    # Use max-tracker (no break) because tl.checkpoints is sorted by t_local,
    # not t_global, so ordering is not guaranteed.
    latest_global = None
    for c in tl.checkpoints:
        if c.t_global <= t and c.response_global_json:
            if latest_global is None or c.t_global > latest_global.t_global:
                latest_global = c

    # Diary: synthesize entries live so the panel never lags behind the local
    # and global responses. For each checkpoint whose local response has landed
    # by time t, append "Steps ~N [pose]: <response_local>". For each whose
    # global response has landed and is non-empty, append the "Checkpoint ~N"
    # line. Sort by landing time so the diary reads chronologically.
    diary_events = []  # list of (landing_time, kind_str, text_str)
    for c in tl.checkpoints:
        if c.t_local <= t:
            pose = _pose_for_checkpoint(tl, c.subgoal_index, c.checkpoint_step)
            line = f"Steps ~{c.checkpoint_step} {_format_pose(pose)}: {c.response_local_text}"
            diary_events.append((c.t_local, "step", line))
        if c.t_global <= t and c.response_global_json:
            completion = c.response_global_json.get("completion_percentage", 0)
            line = f"Checkpoint ~{c.checkpoint_step}: completion = {completion:.2f}"
            diary_events.append((c.t_global, "checkpoint", line))
    diary_events.sort(key=lambda x: x[0])
    diary_entries = [{"kind": kind, "text": text} for _, kind, text in diary_events]

    # Sticky-latest convergence across all subgoals.
    # tl.convergences is sorted by t, so ordered scan is fine.
    latest_conv = None
    for cv in tl.convergences:
        if cv.t <= t:
            latest_conv = cv
        else:
            break

    # Topbar completion + drive: from latest_global.
    completion = 0.0
    drive = "stop"
    if latest_global is not None:
        completion = float(latest_global.response_global_json.get("completion_percentage", 0))
        drive = latest_global.response_global_json.get("drive_action", "stop")

    state = {
        "bar_time": _format_bar_time(t, tl.duration_s),
        "subgoal_index": active_sg,
        "subgoal_total": max(1, len(tl.subgoal_texts) - 1),
        "subgoal_slug": sg_slug,
        "subgoal_text": sg_text,
        "completion": completion,
        "drive": drive,

        "local_image": str(latest_local.grid_local_path) if latest_local else None,
        "local_reasoning": latest_local.response_local_text if latest_local else None,

        "global_image": str(latest_global.grid_global_path) if latest_global else None,
        "global_reasoning": latest_global.response_global_json.get("reasoning") if latest_global else None,
        "global_chips": _global_chips(latest_global.response_global_json) if latest_global else [],

        "diary_entries": diary_entries,

        "conv_status": (
            "complete" if latest_conv and latest_conv.response_json.get("complete")
            else (latest_conv.response_json.get("diagnosis") if latest_conv else None)
        ),
        "conv_reasoning": latest_conv.response_json.get("reasoning") if latest_conv else None,
        "conv_chips": _conv_chips(latest_conv.response_json, latest_conv.subgoal_index) if latest_conv else [],
    }
    return state


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "state":
        tl = load(Path(sys.argv[2]))
        for t_str in sys.argv[3:]:
            t = float(t_str)
            print(f"--- state at t={t:.1f} ---")
            for k, v in state_at(tl, t).items():
                if k in ("local_image", "global_image"):
                    print(f"  {k}: {v}")
                elif k == "diary_entries":
                    print(f"  diary_entries: {len(v)} entries")
                elif k == "global_chips" or k == "conv_chips":
                    print(f"  {k}: {[c['label'] for c in v]}")
                else:
                    s = repr(v)
                    if len(s) > 90: s = s[:87] + "..."
                    print(f"  {k}: {s}")
    else:
        main()
