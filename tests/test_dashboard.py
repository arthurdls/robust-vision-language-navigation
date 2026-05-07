"""Tests for rvln.mininav.dashboard -- web dashboard backend."""

import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from rvln.mininav.dashboard import RunState, build_run_state


def _make_artifacts(run_dir: Path, subgoal_idx: int, subgoal_name: str,
                    checkpoint_idx: int, *,
                    diary: str = "", response: str = "",
                    local_png: bytes = b"", global_png: bytes = b"",
                    convergence_idx: int | None = None,
                    convergence_response: str = "",
                    convergence_png: bytes = b"") -> Path:
    """Helper: write a synthetic subgoal/checkpoint tree under run_dir."""
    sg_dir = run_dir / f"subgoal_{subgoal_idx:02d}_{subgoal_name}"
    art = sg_dir / "diary_artifacts"
    cp = art / f"checkpoint_{checkpoint_idx:04d}"
    cp.mkdir(parents=True, exist_ok=True)
    (cp / "diary.txt").write_text(diary)
    (cp / "response_global.txt").write_text(response)
    (cp / "grid_local.png").write_bytes(local_png)
    (cp / "grid_global.png").write_bytes(global_png)
    if convergence_idx is not None:
        cv = art / f"convergence_{convergence_idx:03d}"
        cv.mkdir(parents=True, exist_ok=True)
        (cv / "response_00.txt").write_text(convergence_response)
        (cv / "grid_convergence_00.png").write_bytes(convergence_png)
    return sg_dir


class TestBuildRunState:
    def test_empty_run_dir(self, tmp_path):
        state = build_run_state(tmp_path)
        assert state.subgoals == []
        assert state.active_subgoal is None
        assert state.checkpoint_label is None
        assert state.run_complete is False

    def test_single_subgoal_single_checkpoint(self, tmp_path):
        _make_artifacts(tmp_path, 1, "ascend", 10,
                        diary="d", response="r",
                        local_png=b"L", global_png=b"G")
        state = build_run_state(tmp_path)
        assert [s["index"] for s in state.subgoals] == [1]
        assert state.subgoals[0]["name"] == "ascend"
        assert state.active_subgoal == {"index": 1, "name": "ascend"}
        assert state.checkpoint_label == "checkpoint_0010"
        assert state.checkpoint_count == 1
        assert state.diary_text == "d"
        assert state.response_text == "r"
        assert state.local_image_mtime > 0
        assert state.global_image_mtime > 0

    def test_checkpoint_count_in_active_subgoal(self, tmp_path):
        _make_artifacts(tmp_path, 1, "ascend", 10)
        _make_artifacts(tmp_path, 1, "ascend", 20)
        _make_artifacts(tmp_path, 1, "ascend", 30)
        state = build_run_state(tmp_path)
        assert state.checkpoint_count == 3

    def test_active_subgoal_is_one_with_newest_checkpoint(self, tmp_path):
        _make_artifacts(tmp_path, 1, "ascend", 10)
        time.sleep(0.02)
        _make_artifacts(tmp_path, 2, "cruise", 5)
        state = build_run_state(tmp_path)
        assert state.active_subgoal["index"] == 2
        assert state.active_subgoal["name"] == "cruise"

    def test_run_complete_when_run_info_exists(self, tmp_path):
        (tmp_path / "run_info.json").write_text("{}")
        state = build_run_state(tmp_path)
        assert state.run_complete is True

    def test_convergence_panel_populated(self, tmp_path):
        _make_artifacts(tmp_path, 1, "ascend", 10,
                        convergence_idx=2,
                        convergence_response="converged: yes",
                        convergence_png=b"C")
        state = build_run_state(tmp_path)
        assert state.convergence_label == "convergence_002"
        assert state.convergence_response == "converged: yes"
        assert state.convergence_image_mtime > 0


import http.client
import threading
from rvln.mininav.dashboard import (
    DashboardHTTPServer, DashboardRequestHandler, build_run_state,
)


@pytest.fixture
def running_server(tmp_path):
    """Boot a real DashboardHTTPServer on an OS-assigned port."""
    server = DashboardHTTPServer(
        ("127.0.0.1", 0),
        DashboardRequestHandler,
        run_dir=tmp_path,
        snapshot_provider=lambda: build_run_state(tmp_path),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, server.server_address[1], tmp_path
    server.shutdown()
    thread.join(timeout=2.0)


class TestRoutes:
    def _get(self, port, path):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, dict(resp.getheaders()), body

    def test_root_serves_html(self, running_server):
        _, port, _ = running_server
        status, headers, body = self._get(port, "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b"<html" in body.lower() or b"<!doctype" in body.lower()

    def test_state_endpoint_returns_json(self, running_server):
        _, port, _ = running_server
        status, headers, body = self._get(port, "/api/state")
        assert status == 200
        assert headers["Content-Type"] == "application/json"
        payload = json.loads(body)
        assert "subgoals" in payload
        assert "server_time" in payload

    def test_state_endpoint_reflects_disk(self, running_server):
        _, port, run_dir = running_server
        _make_artifacts(run_dir, 1, "ascend", 10, diary="hello")
        status, _, body = self._get(port, "/api/state")
        assert status == 200
        payload = json.loads(body)
        assert payload["diary_text"] == "hello"
        assert payload["active_subgoal"]["name"] == "ascend"

    def test_image_route_returns_png(self, running_server):
        _, port, run_dir = running_server
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        _make_artifacts(run_dir, 1, "ascend", 10, local_png=png_bytes)
        status, headers, body = self._get(port, "/img/local")
        assert status == 200
        assert headers["Content-Type"] == "image/png"
        assert body == png_bytes

    def test_image_route_404_when_missing(self, running_server):
        _, port, _ = running_server
        status, _, _ = self._get(port, "/img/local")
        assert status == 404

    def test_image_route_rejects_unknown_slot(self, running_server):
        _, port, _ = running_server
        status, _, _ = self._get(port, "/img/banana")
        assert status == 404


from unittest.mock import MagicMock, patch
from rvln.mininav.dashboard import find_chromium_executable, launch_browser


class TestBrowserLauncher:
    def test_returns_first_existing(self, tmp_path):
        existing = tmp_path / "chrome.exe"
        existing.write_text("")
        candidates = [tmp_path / "missing.exe", existing, tmp_path / "other.exe"]
        result = find_chromium_executable(candidates)
        assert result == existing

    def test_returns_none_when_no_candidate_exists(self, tmp_path):
        candidates = [tmp_path / "a", tmp_path / "b"]
        assert find_chromium_executable(candidates) is None

    def test_launch_uses_app_flag(self, tmp_path):
        fake_chrome = tmp_path / "chrome.exe"
        fake_chrome.write_text("")
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = "popen-handle"
            handle = launch_browser("http://127.0.0.1:8765", browser=fake_chrome)
            assert handle == "popen-handle"
            args, _ = mock_popen.call_args
            cmd = args[0]
            assert str(fake_chrome) in cmd
            assert any("--app=http://127.0.0.1:8765" in a for a in cmd)

    def test_launch_returns_none_when_no_browser(self):
        with patch("rvln.mininav.dashboard.find_chromium_executable", return_value=None), \
             patch("webbrowser.open") as wo:
            handle = launch_browser("http://127.0.0.1:8765")
            assert handle is None
            wo.assert_called_once_with("http://127.0.0.1:8765")


from rvln.mininav.dashboard import MonitorDashboard


class TestMonitorDashboard:
    def test_start_boots_server_and_launches_browser(self, tmp_path):
        fake_handle = MagicMock()
        with patch("rvln.mininav.dashboard.launch_browser", return_value=fake_handle) as launch:
            dash = MonitorDashboard(run_dir=tmp_path, poll_interval_s=0.05)
            dash.start()
            try:
                # Hit the live server on its bound port to prove it's up.
                port = dash.port
                assert port is not None
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
                conn.request("GET", "/api/state")
                resp = conn.getresponse()
                assert resp.status == 200
                conn.close()
            finally:
                dash.close()
            launch.assert_called_once()

    def test_close_terminates_browser(self, tmp_path):
        fake_handle = MagicMock()
        with patch("rvln.mininav.dashboard.launch_browser", return_value=fake_handle):
            dash = MonitorDashboard(run_dir=tmp_path, poll_interval_s=0.05)
            dash.start()
            dash.close()
            fake_handle.terminate.assert_called_once()

    def test_close_is_idempotent(self, tmp_path):
        with patch("rvln.mininav.dashboard.launch_browser", return_value=None):
            dash = MonitorDashboard(run_dir=tmp_path, poll_interval_s=0.05)
            dash.start()
            dash.close()
            dash.close()  # second call should be a no-op, not raise

    def test_start_swallows_browser_failure(self, tmp_path):
        with patch("rvln.mininav.dashboard.launch_browser",
                   side_effect=RuntimeError("boom")):
            dash = MonitorDashboard(run_dir=tmp_path, poll_interval_s=0.05)
            dash.start()  # must not raise; run continues without dashboard
            dash.close()
