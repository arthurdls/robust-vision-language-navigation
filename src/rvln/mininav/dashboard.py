"""Live operator web dashboard for run_hardware.

Pops up a chromeless Chrome / Edge window during a run that shows:
  - a header strip (subgoal index/name, elapsed time, checkpoint
    count, status pill)
  - a subgoal breadcrumb across the top
  - the last checkpoint's local + global prompt images
  - the diary
  - the last global response
  - the last convergence event (image + response)

Architecture:
- Runs an http.server.ThreadingHTTPServer on a daemon thread inside
  the run process. Binds to 127.0.0.1 on a free port.
- Spawns the browser as a child process via Chrome/Edge --app= mode
  so we can terminate it cleanly on shutdown.
- The server polls run_dir on its own thread (4 Hz) and caches a
  RunState snapshot under a lock. HTTP handlers read snapshots; they
  never touch disk directly.
- close() shuts the server, kills the browser, joins the thread.

The dashboard is best-effort: any failure (port bind, browser
launch, file read) logs a warning and the run proceeds.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Section 1: RunState snapshot
# ----------------------------------------------------------------------


@dataclass
class RunState:
    """Snapshot of everything the client needs to render one frame.

    Built by build_run_state() at 4 Hz from disk; served as JSON via
    /api/state. Mtimes double as cache-busting query params for the
    image routes (the client only re-fetches /img/local when its
    mtime changes).
    """
    subgoals: list = field(default_factory=list)
    active_subgoal: Optional[dict] = None
    checkpoint_label: Optional[str] = None
    checkpoint_count: int = 0
    diary_text: str = ""
    diary_mtime: float = 0.0
    response_text: str = ""
    response_mtime: float = 0.0
    local_image_mtime: float = 0.0
    global_image_mtime: float = 0.0
    convergence_label: Optional[str] = None
    convergence_response: str = ""
    convergence_image_mtime: float = 0.0
    run_complete: bool = False
    server_time: float = 0.0


def _parse_subgoal_dir(p: Path) -> Optional[dict]:
    """subgoal_03_descend_to_the_ground -> {"index": 3, "name": "descend_to_the_ground"}."""
    parts = p.name.split("_", 2)
    if len(parts) < 3 or parts[0] != "subgoal":
        return None
    try:
        idx = int(parts[1])
    except ValueError:
        return None
    return {"index": idx, "name": parts[2]}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def build_run_state(run_dir: Path) -> RunState:
    """Read run_dir from disk and return a RunState snapshot.

    Globbing across all subgoals (not just the active one) means the
    previous subgoal's last checkpoint stays visible until the next
    subgoal produces its first one.
    """
    state = RunState(server_time=time.time())
    state.run_complete = (run_dir / "run_info.json").exists()

    sg_dirs = sorted(
        (p for p in run_dir.glob("subgoal_*") if p.is_dir()),
        key=lambda p: p.name,
    )
    for sg in sg_dirs:
        info = _parse_subgoal_dir(sg)
        if info is not None:
            state.subgoals.append(info)

    cp_glob = list(run_dir.glob("subgoal_*/diary_artifacts/checkpoint_*"))
    if cp_glob:
        cp = max(cp_glob, key=lambda p: p.stat().st_mtime)
        state.checkpoint_label = cp.name
        state.active_subgoal = _parse_subgoal_dir(cp.parent.parent)
        state.checkpoint_count = sum(
            1 for _ in cp.parent.glob("checkpoint_*")
        )
        state.diary_text = _read_text(cp / "diary.txt")
        state.diary_mtime = _mtime(cp / "diary.txt")
        state.response_text = _read_text(cp / "response_global.txt")
        state.response_mtime = _mtime(cp / "response_global.txt")
        state.local_image_mtime = _mtime(cp / "grid_local.png")
        state.global_image_mtime = _mtime(cp / "grid_global.png")

    cv_glob = list(run_dir.glob("subgoal_*/diary_artifacts/convergence_*"))
    if cv_glob:
        cv = max(cv_glob, key=lambda p: p.stat().st_mtime)
        state.convergence_label = cv.name
        responses = sorted(cv.glob("response_*.txt"))
        if responses:
            state.convergence_response = _read_text(responses[-1])
        images = sorted(cv.glob("grid_convergence_*.png"))
        if images:
            state.convergence_image_mtime = _mtime(images[-1])
    return state


def state_to_json(state: RunState) -> bytes:
    return json.dumps(asdict(state)).encode("utf-8")


# ----------------------------------------------------------------------
# Section 2: HTTP server
# ----------------------------------------------------------------------

import http.server
import socketserver
from typing import Callable

_STATIC_DIR = Path(__file__).resolve().parent / "dashboard_static"
_IMG_FILES = {
    "local":       lambda cp: cp / "grid_local.png",
    "global":      lambda cp: cp / "grid_global.png",
    "convergence": lambda cv: cv,
}


class DashboardHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """ThreadingHTTPServer that carries the run_dir + snapshot provider."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler_cls, *,
                 run_dir: Path, snapshot_provider: Callable[[], RunState]):
        super().__init__(addr, handler_cls)
        self.run_dir = Path(run_dir)
        self.snapshot_provider = snapshot_provider


class DashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    """Routes: / , /static/*, /api/state, /img/{local,global,convergence}."""

    def log_message(self, format, *args):
        # Silence the default per-request stderr log; it spams during 4 Hz polling.
        pass

    def do_GET(self):
        try:
            if self.path == "/" or self.path == "/index.html":
                self._serve_static("index.html", "text/html; charset=utf-8")
            elif self.path.startswith("/static/"):
                name = self.path[len("/static/"):].split("?", 1)[0]
                self._serve_static(name, self._mime_for(name))
            elif self.path.split("?", 1)[0] == "/api/state":
                self._serve_state()
            elif self.path.startswith("/img/"):
                slot = self.path[len("/img/"):].split("?", 1)[0]
                self._serve_image(slot)
            else:
                self._send_simple(404, b"not found")
        except Exception as exc:
            logger.debug("dashboard handler error on %s: %s", self.path, exc)
            self._send_simple(500, b"server error")

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _mime_for(name: str) -> str:
        if name.endswith(".css"):
            return "text/css; charset=utf-8"
        if name.endswith(".js"):
            return "application/javascript; charset=utf-8"
        if name.endswith(".html"):
            return "text/html; charset=utf-8"
        return "application/octet-stream"

    def _serve_static(self, name: str, mime: str) -> None:
        path = _STATIC_DIR / name
        if ".." in name or not path.is_file():
            self._send_simple(404, b"not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_state(self) -> None:
        state = self.server.snapshot_provider()
        body = state_to_json(state)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self, slot: str) -> None:
        if slot not in _IMG_FILES:
            self._send_simple(404, b"unknown slot")
            return
        path = _resolve_image_path(self.server.run_dir, slot)
        if path is None or not path.is_file():
            self._send_simple(404, b"no image yet")
            return
        try:
            body = path.read_bytes()
        except Exception:
            self._send_simple(500, b"image read failed")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_simple(self, status: int, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            pass


def _resolve_image_path(run_dir: Path, slot: str) -> Optional[Path]:
    """Look up the latest checkpoint or convergence dir, then the right PNG."""
    if slot == "convergence":
        cv_glob = list(run_dir.glob("subgoal_*/diary_artifacts/convergence_*"))
        if not cv_glob:
            return None
        cv = max(cv_glob, key=lambda p: p.stat().st_mtime)
        images = sorted(cv.glob("grid_convergence_*.png"))
        return images[-1] if images else None
    cp_glob = list(run_dir.glob("subgoal_*/diary_artifacts/checkpoint_*"))
    if not cp_glob:
        return None
    cp = max(cp_glob, key=lambda p: p.stat().st_mtime)
    name = "grid_local.png" if slot == "local" else "grid_global.png"
    return cp / name


# ----------------------------------------------------------------------
# Section 3: Browser launcher
# ----------------------------------------------------------------------

import platform
import subprocess
import webbrowser

WINDOW_SIZE = "1400,900"


def _wsl_chromium_candidates() -> list[Path]:
    return [
        Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path("/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ]


def _linux_chromium_candidates() -> list[Path]:
    return [
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/chromium"),
        Path("/usr/bin/chromium-browser"),
        Path("/usr/bin/microsoft-edge"),
    ]


def _macos_chromium_candidates() -> list[Path]:
    return [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ]


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def _platform_candidates() -> list[Path]:
    if _is_wsl():
        return _wsl_chromium_candidates()
    if platform.system() == "Darwin":
        return _macos_chromium_candidates()
    return _linux_chromium_candidates()


def find_chromium_executable(candidates: Optional[list[Path]] = None) -> Optional[Path]:
    """Return the first candidate path that exists, or None."""
    for c in candidates if candidates is not None else _platform_candidates():
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def launch_browser(url: str, browser: Optional[Path] = None) -> Optional[subprocess.Popen]:
    """Launch the URL in a chromeless --app window.

    Returns the Popen handle on success so the caller can terminate
    the browser when the run ends. If no Chromium-family browser is
    found, falls back to webbrowser.open and returns None (auto-close
    not available in that case).
    """
    chrome = browser if browser is not None else find_chromium_executable()
    if chrome is None:
        logger.warning(
            "Dashboard: no Chrome / Edge found; opening default browser. "
            "Auto-close on run end will not work; close the tab manually.",
        )
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Dashboard: webbrowser.open failed: %s", exc)
        return None
    cmd = [
        str(chrome),
        f"--app={url}",
        f"--window-size={WINDOW_SIZE}",
        "--new-window",
    ]
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        logger.warning("Dashboard: failed to launch %s: %s", chrome, exc)
        return None


# ----------------------------------------------------------------------
# Section 4: MonitorDashboard (public lifecycle)
# ----------------------------------------------------------------------


class MonitorDashboard:
    """Boots the HTTP server + browser; tears them down on close().

    Public interface matches the old Tkinter dashboard so interface.py
    requires no changes:
      MonitorDashboard(run_dir).start() / .close()
    """

    DEFAULT_PORT = 8765
    BROWSER_KILL_TIMEOUT_S = 2.0

    def __init__(self, run_dir: Path, poll_interval_s: float = 0.25):
        self.run_dir = Path(run_dir)
        self.poll_interval_s = float(poll_interval_s)
        self.port: Optional[int] = None
        self._server: Optional[DashboardHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._browser: Optional[subprocess.Popen] = None
        self._poller_stop = threading.Event()
        self._poller_thread: Optional[threading.Thread] = None
        self._snapshot_lock = threading.Lock()
        self._snapshot = RunState()
        self._closed = False

    def start(self) -> None:
        try:
            self._start_server()
            self._start_poller()
            self._launch_browser()
        except Exception as exc:
            logger.warning("Dashboard: failed to start (%s); run continues.", exc)
            self._safe_teardown()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._safe_teardown()

    # -- internal --------------------------------------------------------

    def _start_server(self) -> None:
        # Try DEFAULT_PORT first for predictable URLs; fall back to OS-assigned.
        for addr in [("127.0.0.1", self.DEFAULT_PORT), ("127.0.0.1", 0)]:
            try:
                self._server = DashboardHTTPServer(
                    addr, DashboardRequestHandler,
                    run_dir=self.run_dir,
                    snapshot_provider=self._get_snapshot,
                )
                break
            except OSError:
                continue
        if self._server is None:
            raise RuntimeError("could not bind dashboard server to 127.0.0.1")
        self.port = self._server.server_address[1]
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="dashboard-http", daemon=True,
        )
        self._server_thread.start()
        logger.info("Dashboard: serving on http://127.0.0.1:%d", self.port)

    def _start_poller(self) -> None:
        self._poller_stop.clear()
        self._poller_thread = threading.Thread(
            target=self._poll_loop, name="dashboard-poll", daemon=True,
        )
        self._poller_thread.start()

    def _poll_loop(self) -> None:
        while not self._poller_stop.is_set():
            try:
                snap = build_run_state(self.run_dir)
                with self._snapshot_lock:
                    self._snapshot = snap
            except Exception:
                logger.exception("Dashboard: snapshot build failed; retrying.")
            self._poller_stop.wait(self.poll_interval_s)

    def _get_snapshot(self) -> RunState:
        with self._snapshot_lock:
            return self._snapshot

    def _launch_browser(self) -> None:
        try:
            self._browser = launch_browser(f"http://127.0.0.1:{self.port}/")
        except Exception as exc:
            logger.warning("Dashboard: launch_browser raised (%s); skipping.", exc)
            self._browser = None

    def _safe_teardown(self) -> None:
        if self._browser is not None:
            try:
                self._browser.terminate()
                try:
                    self._browser.wait(timeout=self.BROWSER_KILL_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    self._browser.kill()
            except (ProcessLookupError, OSError):
                pass
            except Exception as exc:
                logger.debug("Dashboard: browser terminate raised %s", exc)
        self._browser = None

        self._poller_stop.set()
        if self._poller_thread is not None and self._poller_thread.is_alive():
            self._poller_thread.join(timeout=1.0)
        self._poller_thread = None

        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception as exc:
                logger.debug("Dashboard: server shutdown raised %s", exc)
        self._server = None
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)
        self._server_thread = None
