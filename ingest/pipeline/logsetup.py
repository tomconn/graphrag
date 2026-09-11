"""Logging setup with dead-pipe resilience and a stall watchdog.

Two failure modes of a plain ``logging.basicConfig`` motivated this module:

* A one-off ``docker compose run`` whose attached client dies (e.g. the
  wrapper process is killed) leaves the container running with a broken
  stdout pipe. A plain StreamHandler then blocks forever on its next write,
  wedging every worker thread — the container looks alive but nothing
  progresses. Mitigation: stdout is switched to non-blocking and this
  handler drops records instead of blocking on a full/broken pipe; every
  record is also written to a file (``INGEST_LOG_FILE``, default
  ``/app/logs/ingest.log``, keep it on a bind mount) so the run remains
  diagnosable even when stdout is dead.
* If the pipeline genuinely stops making progress, the process would
  otherwise sit silently forever. The watchdog thread warns after
  ``INGEST_STALL_SECONDS`` (default 600) without a log record and
  force-exits with code 75 after a second consecutive stall window, so the
  one-off container terminates visibly instead of hanging.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

STALL_EXIT_CODE = 75


class DropOnBrokenPipeStreamHandler(logging.StreamHandler):
    """StreamHandler that drops a record rather than blocking forever when
    the underlying stream is a full or broken pipe (dead docker client).
    """

    def __init__(self, stream):
        super().__init__(stream)
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except (BlockingIOError, BrokenPipeError, OSError, ValueError):
            self.dropped += 1
        except Exception:  # noqa: BLE001 — logging must never raise
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        # The default handleError writes to stderr — which may be the same
        # dead pipe. Swallow instead.
        pass


class ActivityFilter(logging.Filter):
    """Records the timestamp of the last record seen; attach to a root
    handler so propagated records from every child logger count.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last = time.monotonic()

    def filter(self, record: logging.LogRecord) -> bool:
        self.last = time.monotonic()
        return True


def _watchdog(activity: ActivityFilter, stall_seconds: int,
              poll_seconds: float) -> None:
    """Warn once after a stall window; force-exit after two consecutive
    windows with no log activity at all.
    """
    log = logging.getLogger("ingest.watchdog")
    warned = False
    while True:
        time.sleep(poll_seconds)
        elapsed = time.monotonic() - activity.last
        if elapsed >= 2 * stall_seconds:
            log.error("No log activity for %.0fs — ingest appears wedged; "
                      "forcing exit %d", elapsed, STALL_EXIT_CODE)
            os._exit(STALL_EXIT_CODE)
        if elapsed >= stall_seconds and not warned:
            log.warning("No log activity for %.0fs (threshold %ds) — if this "
                        "persists, the watchdog will force-exit the ingest "
                        "container", elapsed, stall_seconds)
            warned = True
        if elapsed < stall_seconds:
            warned = False


def setup_logging(stall_seconds: int | None = None,
                  poll_seconds: float | None = None) -> None:
    """Configure logging: non-blocking stdout + a file sink + the watchdog.

    Returns the ActivityFilter the watchdog uses, for tests. Also starts the
    watchdog daemon thread unless stall_seconds is 0 (disabled).
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Non-blocking stdout: a full/broken pipe must never wedge the pipeline.
    try:
        os.set_blocking(sys.stdout.fileno(), False)
    except (OSError, ValueError, AttributeError):  # not a real fd (tests, tty)
        pass
    stream_handler = DropOnBrokenPipeStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    # File sink: survives a dead stdout so a wedged or interrupted run stays
    # diagnosable via the bind mount.
    activity = ActivityFilter()
    stream_handler.addFilter(activity)  # watchdog tracks records that pass here
    log_file = os.environ.get("INGEST_LOG_FILE", "/app/logs/ingest.log")
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        file_handler.addFilter(activity)
        root.addHandler(file_handler)
    except OSError as exc:
        stream_handler.emit(logging.LogRecord(
            "ingest.watchdog", logging.WARNING, __file__, 0,
            "Could not open ingest log file %s: %s", (log_file, exc), None))

    stall = int(os.environ.get("INGEST_STALL_SECONDS", "600")
                if stall_seconds is None else stall_seconds)
    if stall > 0:
        poll = poll_seconds if poll_seconds is not None else max(1.0, stall / 10)
        threading.Thread(target=_watchdog, args=(activity, stall, poll),
                         daemon=True, name="ingest-watchdog").start()
    return activity