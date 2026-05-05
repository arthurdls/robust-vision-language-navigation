"""Live operator dashboard for run_hardware.

Pops up a Tkinter window during a run that shows:
  - the last checkpoint's local prompt image (the prev/curr frame pair)
  - the last checkpoint's global prompt image (the temporally-sampled grid)
  - the diary (every entry)
  - the last global VLM response
  - the last convergence VLM exchange (latest prompt + response, plus all
    earlier retries from the same convergence event)

Each panel shows a wall-clock timestamp pulled from the artifact file's
mtime so the operator can correlate output to events.

Architecture:
- Runs on its own daemon thread with its own Tk root and mainloop, so it
  never blocks the control loop.
- Polls run_dir on a tk.after schedule (default once per second). The
  monitor writes artifacts directly to disk (`_save_checkpoint_artifact`,
  `_save_convergence_artifact`); the dashboard never touches the live
  monitor object, so there is zero shared state and no thread-safety
  concerns.
- close() flips a stop event and schedules root.quit() onto the Tk
  thread so shutdown is cooperative even though the mainloop is on a
  different thread.

The dashboard is best-effort: if Tkinter or Pillow isn't available, or
the display server isn't reachable (headless run), start() logs a
warning and the run proceeds without the popup.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MonitorDashboard:
    """Polls a run_dir for monitor artifacts and renders them in Tkinter."""

    # Image display sizes are computed at UI build time from the actual
    # screen dimensions (see _build_ui), so a dashboard on a 1920x1080
    # monitor uses much bigger panels than one on a 1366x768 laptop. The
    # values below are only fallbacks if Tk can't probe the screen.
    LOCAL_IMG_W = 720
    LOCAL_IMG_H = 405
    GLOBAL_IMG_W = 720
    GLOBAL_IMG_H = 405

    # Font sizes. Operator feedback: still hard to read at 13pt sharing
    # screen with the run terminal. Bumped again to 17pt body so the
    # dashboard is readable from a meter or two away.
    TEXT_FONT = ("Courier", 17)
    HEADER_FONT = ("Helvetica", 16, "bold")
    TIMESTAMP_FONT = ("Helvetica", 14)

    # Fraction of the screen the window covers (92%). Leaves room for the
    # window manager's titlebar / taskbar so nothing is clipped.
    SCREEN_FRACTION = 0.92

    def __init__(self, run_dir: Path, poll_interval_s: float = 0.25):
        self.run_dir = Path(run_dir)
        self.poll_interval_s = float(poll_interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available = self._probe_available()

        # Cached snapshot of what's currently rendered, so we only push
        # bytes to Tk when something actually changed (resizing PIL images
        # is expensive).
        self._cur_local_path: Optional[Path] = None
        self._cur_local_mtime: Optional[float] = None
        self._cur_global_path: Optional[Path] = None
        self._cur_global_mtime: Optional[float] = None
        self._cur_diary_text: Optional[str] = None
        self._cur_response_text: Optional[str] = None
        self._cur_conv_text: Optional[str] = None

        # Tk handles created on the dashboard thread.
        self._root = None
        self._local_label = None
        self._global_label = None
        self._diary_text = None
        self._response_text = None
        self._conv_text = None
        self._local_ts_var = None
        self._global_ts_var = None
        self._diary_ts_var = None
        self._response_ts_var = None
        self._conv_ts_var = None
        self._local_photo = None
        self._global_photo = None

    @staticmethod
    def _probe_available() -> bool:
        """Verify Tkinter + Pillow are importable. Display reachability is
        only checked when the thread actually tries to construct Tk()."""
        try:
            import tkinter  # noqa: F401
            from PIL import Image, ImageTk  # noqa: F401
        except Exception as exc:
            logger.warning(
                "MonitorDashboard: Tkinter or Pillow not importable (%s); "
                "live dashboard disabled.", exc,
            )
            return False
        return True

    def start(self) -> None:
        """Spawn the dashboard thread. No-op if Tkinter import failed."""
        if not self._available:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_thread, name="dashboard", daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Tear down the dashboard. Schedules quit() on the Tk thread."""
        self._stop.set()
        if self._root is not None:
            try:
                self._root.after(0, self._root.quit)
            except Exception:
                # Root may already be destroyed; ignore.
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("MonitorDashboard thread did not exit within 2s.")

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run_thread(self) -> None:
        try:
            self._build_ui()
        except Exception as exc:
            logger.warning(
                "MonitorDashboard: failed to open Tk window (%s); the run "
                "will proceed without a live dashboard.", exc,
            )
            return
        # Schedule the first refresh, then enter the Tk mainloop.
        self._root.after(int(self.poll_interval_s * 1000), self._refresh_tick)
        try:
            self._root.mainloop()
        except Exception:
            logger.exception("MonitorDashboard mainloop crashed")

    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._root = tk.Tk()
        self._root.title("MiniNav Live Monitor")

        # Size the window to fit the actual screen. Operator runs the
        # dashboard alongside the run terminal on a single monitor, so
        # we want it big enough to be readable but not so big it covers
        # the taskbar.
        try:
            sw = self._root.winfo_screenwidth()
            sh = self._root.winfo_screenheight()
            w = max(800, int(sw * self.SCREEN_FRACTION))
            h = max(600, int(sh * self.SCREEN_FRACTION))
            self._root.geometry(f"{w}x{h}")
            # Recompute image panel sizes to use ~42% of window width per
            # image at 16:9 ratio. Big monitors get big panels.
            img_w = max(480, int(w * 0.42))
            img_h = int(img_w * 9 / 16)
            self.LOCAL_IMG_W = self.GLOBAL_IMG_W = img_w
            self.LOCAL_IMG_H = self.GLOBAL_IMG_H = img_h
        except Exception:
            # Fall back to the class-level defaults if Tk can't probe
            # screen dims (rare; happens on some headless display
            # configurations).
            self._root.geometry("1280x900")

        # Make ttk.LabelFrame titles match the operator-readable size we
        # use for body text. The default ttk theme renders these tiny.
        try:
            style = ttk.Style()
            style.configure("TLabelframe.Label", font=self.HEADER_FONT)
            # Bump scrollbar arrow widgets so they're easier to grab on
            # high-DPI displays.
            style.configure("Vertical.TScrollbar", arrowsize=18)
        except Exception:
            pass

        # 3 rows, 2 columns. Top row = images, middle = text panels,
        # bottom = full-width convergence panel. Configure weights so
        # the text rows take up remaining space when the window is
        # resized.
        for col in range(2):
            self._root.grid_columnconfigure(col, weight=1)
        self._root.grid_rowconfigure(1, weight=1)
        self._root.grid_rowconfigure(2, weight=1)

        # Row 0: local prompt image (left), global prompt image (right).
        self._local_label, self._local_ts_var = self._make_image_panel(
            parent=self._root, row=0, column=0,
            title="Last checkpoint: local prompt (prev / curr)",
        )
        self._global_label, self._global_ts_var = self._make_image_panel(
            parent=self._root, row=0, column=1,
            title="Last checkpoint: global prompt (sampled grid)",
        )

        # Row 1: diary (left), last global response (right).
        self._diary_text, self._diary_ts_var = self._make_text_panel(
            parent=self._root, row=1, column=0, title="Diary",
        )
        self._response_text, self._response_ts_var = self._make_text_panel(
            parent=self._root, row=1, column=1, title="Last global response",
        )

        # Row 2: convergence panel spans both columns.
        self._conv_text, self._conv_ts_var = self._make_text_panel(
            parent=self._root, row=2, column=0, columnspan=2,
            title="Last convergence (all retries from the latest event)",
        )

    def _make_image_panel(self, parent, row, column, title):
        import tkinter as tk
        from tkinter import ttk

        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=row, column=column, padx=6, pady=6, sticky="nsew")
        ts_var = tk.StringVar(value="(waiting for first checkpoint)")
        ts = ttk.Label(
            frame, textvariable=ts_var, foreground="gray30",
            font=self.TIMESTAMP_FONT,
        )
        ts.pack(anchor="w", padx=4)
        img_label = ttk.Label(frame)
        img_label.pack(padx=4, pady=4)
        return img_label, ts_var

    def _make_text_panel(self, parent, row, column, title, columnspan=1):
        import tkinter as tk
        from tkinter import ttk

        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(
            row=row, column=column, columnspan=columnspan,
            padx=6, pady=6, sticky="nsew",
        )
        ts_var = tk.StringVar(value="(no event yet)")
        ts = ttk.Label(
            frame, textvariable=ts_var, foreground="gray30",
            font=self.TIMESTAMP_FONT,
        )
        ts.pack(anchor="w", padx=4)

        body = tk.Frame(frame)
        body.pack(fill="both", expand=True, padx=4, pady=4)
        scrollbar = ttk.Scrollbar(body, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        text = tk.Text(
            body, wrap="word", height=8,
            font=self.TEXT_FONT,
            yscrollcommand=scrollbar.set,
        )
        text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text.yview)
        return text, ts_var

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_tick(self) -> None:
        if self._stop.is_set():
            try:
                self._root.quit()
            except Exception:
                pass
            return
        try:
            self._refresh_once()
        except Exception:
            logger.exception("Dashboard refresh failed; will retry next tick")
        if not self._stop.is_set():
            self._root.after(int(self.poll_interval_s * 1000), self._refresh_tick)

    def _refresh_once(self) -> None:
        cp_dir, conv_dir = self._latest_artifact_dirs()
        if cp_dir is not None:
            self._refresh_checkpoint(cp_dir)
        if conv_dir is not None:
            self._refresh_convergence(conv_dir)

    def _latest_artifact_dirs(self):
        """Find the newest checkpoint/convergence dirs anywhere under run_dir.

        Globs across ALL subgoals (not just the current one) so when a
        subgoal completes the last checkpoint of the previous subgoal
        keeps showing until the next subgoal produces its first one.
        """
        cp_glob = list(self.run_dir.glob("subgoal_*/diary_artifacts/checkpoint_*"))
        conv_glob = list(self.run_dir.glob("subgoal_*/diary_artifacts/convergence_*"))
        cp_dir = max(cp_glob, key=lambda p: p.stat().st_mtime) if cp_glob else None
        conv_dir = max(conv_glob, key=lambda p: p.stat().st_mtime) if conv_glob else None
        return cp_dir, conv_dir

    def _refresh_checkpoint(self, cp_dir: Path) -> None:
        # Images: refresh only when path or mtime changed (resizing PIL
        # images is non-trivial).
        local_path = cp_dir / "grid_local.png"
        if local_path.exists():
            mtime = local_path.stat().st_mtime
            if (local_path != self._cur_local_path or
                    mtime != self._cur_local_mtime):
                self._set_image(self._local_label, local_path,
                                self.LOCAL_IMG_W, self.LOCAL_IMG_H,
                                slot="local")
                self._cur_local_path = local_path
                self._cur_local_mtime = mtime
                self._local_ts_var.set(self._format_ts(mtime, cp_dir.name))

        global_path = cp_dir / "grid_global.png"
        if global_path.exists():
            mtime = global_path.stat().st_mtime
            if (global_path != self._cur_global_path or
                    mtime != self._cur_global_mtime):
                self._set_image(self._global_label, global_path,
                                self.GLOBAL_IMG_W, self.GLOBAL_IMG_H,
                                slot="global")
                self._cur_global_path = global_path
                self._cur_global_mtime = mtime
                self._global_ts_var.set(self._format_ts(mtime, cp_dir.name))

        # Diary
        diary_path = cp_dir / "diary.txt"
        if diary_path.exists():
            content = self._read_text(diary_path)
            if content != self._cur_diary_text:
                self._set_text(self._diary_text, content)
                self._cur_diary_text = content
                self._diary_ts_var.set(
                    self._format_ts(diary_path.stat().st_mtime, cp_dir.name),
                )

        # Last global response
        resp_path = cp_dir / "response_global.txt"
        if resp_path.exists():
            content = self._read_text(resp_path)
            if content != self._cur_response_text:
                self._set_text(self._response_text, content)
                self._cur_response_text = content
                self._response_ts_var.set(
                    self._format_ts(resp_path.stat().st_mtime, cp_dir.name),
                )

    def _refresh_convergence(self, conv_dir: Path) -> None:
        # A convergence event can have multiple retries (initial,
        # parse-failure retry, missing-instruction retry). Render every
        # prompt/response pair we find in the directory in order.
        prompts = sorted(conv_dir.glob("prompt_*.txt"))
        responses = sorted(conv_dir.glob("response_*.txt"))
        if not prompts and not responses:
            return
        chunks = []
        latest_mtime = 0.0
        for i, resp_path in enumerate(responses):
            mtime = resp_path.stat().st_mtime
            latest_mtime = max(latest_mtime, mtime)
            ts = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
            chunks.append(f"--- response[{i:02d}] @ {ts} ---")
            chunks.append(self._read_text(resp_path).rstrip())
            chunks.append("")
        rendered = "\n".join(chunks).rstrip()
        if rendered != self._cur_conv_text:
            self._set_text(self._conv_text, rendered)
            self._cur_conv_text = rendered
            self._conv_ts_var.set(
                self._format_ts(latest_mtime, conv_dir.name)
                if latest_mtime else f"{conv_dir.name} (no responses yet)",
            )

    # ------------------------------------------------------------------
    # Tk helpers
    # ------------------------------------------------------------------

    def _set_image(self, label, path: Path, max_w: int, max_h: int, slot: str) -> None:
        # The monitor writes grid_local.png / grid_global.png via
        # PIL.Image.save(); the dashboard polls that same directory on
        # its own thread, so it can race the writer and read a half-
        # written file. Two defenses:
        #   1. ImageFile.LOAD_TRUNCATED_IMAGES=True so PIL accepts a
        #      partial PNG instead of raising "image file is truncated".
        #   2. The whole load+thumbnail+PhotoImage sequence is wrapped
        #      in try/except. A bad read just defers the refresh to the
        #      next tick (mtime hasn't changed, so we'll hit the same
        #      path again next second when it's fully written).
        # Use a context manager + .copy() so the file handle is closed
        # before PhotoImage materializes its bitmap; otherwise
        # PIL keeps a lazy reference to the file.
        from PIL import Image, ImageFile, ImageTk
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        try:
            with Image.open(path) as raw:
                img = raw.copy()
            img.thumbnail((max_w, max_h))
            photo = ImageTk.PhotoImage(img)
        except Exception as exc:
            # Reset the cached mtime so we retry next tick.
            if slot == "local":
                self._cur_local_mtime = None
            else:
                self._cur_global_mtime = None
            logger.debug("Dashboard: image load failed (%s); will retry.", exc)
            return

        label.configure(image=photo)
        # Keep a reference; otherwise Tk garbage-collects the image.
        if slot == "local":
            self._local_photo = photo
        else:
            self._global_photo = photo

    @staticmethod
    def _set_text(widget, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        # Scroll to the bottom so the latest entry is visible.
        widget.see("end")
        widget.configure(state="disabled")

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(errors="replace")
        except Exception as exc:
            return f"(failed to read {path.name}: {exc})"

    @staticmethod
    def _format_ts(mtime: float, label: str) -> str:
        ts = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        return f"{label} @ {ts}"
