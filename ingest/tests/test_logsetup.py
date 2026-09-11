"""Unit tests for ingest/pipeline/logsetup.py — dead-pipe resilience and
the stall watchdog. No services, no real pipes.
"""
import logging
import threading
import time

import pytest

from pipeline import logsetup
from pipeline.logsetup import (ActivityFilter, DropOnBrokenPipeStreamHandler,
                               _watchdog, setup_logging)


class _BrokenPipeStream:
    """A stream that behaves like a dead docker client pipe: every write
    raises BlockingIOError (non-blocking fd with a full buffer)."""

    def __init__(self):
        self.write_calls = 0

    def write(self, text):
        self.write_calls += 1
        raise BlockingIOError()

    def flush(self):
        pass


def _record(msg="hi"):
    return logging.LogRecord("ingest.test", logging.INFO, __file__, 1, msg,
                             (), None)


# ------------------------------------------------- dead-pipe resilience

def test_broken_pipe_stream_handler_drops_instead_of_raising_or_blocking():
    stream = _BrokenPipeStream()
    handler = DropOnBrokenPipeStreamHandler(stream)
    handler.emit(_record())          # must not raise and must not block
    assert stream.write_calls == 1


def test_activity_filter_tracks_last_record_time():
    activity = ActivityFilter()
    before = activity.last
    time.sleep(0.01)
    assert activity.filter(_record()) is True
    assert activity.last > before


# ------------------------------------------------------------ setup_logging

@pytest.fixture
def clean_root_logger():
    before = logging.root.handlers[:]
    before_level = logging.root.level
    yield
    logging.root.handlers = before
    logging.root.setLevel(before_level)


def test_setup_logging_writes_to_file_sink(tmp_path, clean_root_logger,
                                           monkeypatch):
    monkeypatch.setenv("INGEST_LOG_FILE", str(tmp_path / "logs" / "run.log"))
    monkeypatch.setenv("INGEST_STALL_SECONDS", "0")   # no watchdog in tests
    setup_logging()
    logging.getLogger("ingest.test").info("hello sink")
    for handler in logging.root.handlers:
        handler.flush()
    assert (tmp_path / "logs" / "run.log").read_text().endswith("hello sink\n")


def test_setup_logging_starts_watchdog_thread(tmp_path, clean_root_logger,
                                              monkeypatch):
    monkeypatch.setenv("INGEST_LOG_FILE", str(tmp_path / "run.log"))
    monkeypatch.setenv("INGEST_STALL_SECONDS", "0")
    setup_logging()
    assert not any(t.name == "ingest-watchdog" for t in threading.enumerate())
    activity = setup_logging(stall_seconds=3600, poll_seconds=0.05)
    try:
        watchdogs = [t for t in threading.enumerate() if t.name == "ingest-watchdog"]
        assert len(watchdogs) == 1
        assert watchdogs[0].daemon
    finally:
        activity.last = time.monotonic() + 9999  # keep it quiet


# --------------------------------------------------------------- watchdog

def test_watchdog_warns_then_force_exits_after_two_stall_windows(monkeypatch):
    activity = ActivityFilter()
    activity.last = time.monotonic() - 10_000   # wedged since long ago
    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(logsetup.os, "_exit", fake_exit)
    with pytest.raises(SystemExit):
        _watchdog(activity, stall_seconds=1, poll_seconds=0.05)
    assert exits == [logsetup.STALL_EXIT_CODE]


def test_watchdog_stays_quiet_while_active(monkeypatch):
    activity = ActivityFilter()
    activity.last = time.monotonic() + 3600   # never stale
    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(logsetup.os, "_exit", fake_exit)
    thread = threading.Thread(
        target=_watchdog, args=(activity, 3600, 0.05), daemon=True)
    thread.start()
    thread.join(0.3)
    assert thread.is_alive()   # no exit, still polling
    assert exits == []


def test_watchdog_retries_exit_code_distinct_from_fatal_error():
    # exit code 75 must not collide with the fatal-error exit code 1 or
    # the success code 0 (contract in pipeline/main.py).
    assert logsetup.STALL_EXIT_CODE == 75
    assert logsetup.STALL_EXIT_CODE not in (0, 1)