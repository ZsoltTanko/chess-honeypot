"""The log viewer helper: start once, leave a running viewer alone, stop on request."""

import socket

import psutil
import pytest

import viewer


@pytest.fixture
def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_nothing_serves_a_free_port(free_port):
    assert viewer.is_serving(free_port) is False
    assert viewer.stop(port=free_port) is False


def test_ensure_running_starts_once_and_stop_ends_it(tmp_path, free_port):
    url = viewer.ensure_running(tmp_path, free_port)
    try:
        assert url == f"http://127.0.0.1:{free_port}" and viewer.is_serving(free_port)
        pid = int(viewer.pid_file(tmp_path, free_port).read_text())
        assert "view" in psutil.Process(pid).cmdline()
        # a second call finds the running viewer and starts nothing
        assert viewer.ensure_running(tmp_path, free_port) == url
        assert int(viewer.pid_file(tmp_path, free_port).read_text()) == pid
        assert psutil.pid_exists(pid)
    finally:
        stopped = viewer.stop(tmp_path, free_port)
    assert stopped is True
    assert not psutil.pid_exists(pid) and not viewer.is_serving(free_port)
    assert not viewer.pid_file(tmp_path, free_port).exists()
