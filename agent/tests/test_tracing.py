"""Unit tests for agent/app/tracing.py — TraceLogger event file lifecycle,
including the never-break-a-chat swallow rules.
"""
import json
import os

import pytest

from app.tracing import TraceLogger


def read_events(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_start_writes_run_start_event(tmp_path):
    trace = TraceLogger(trace_dir=str(tmp_path / "traces"))
    trace_id = trace.start("What is clause 15?", "hybrid")
    assert trace_id == trace.trace_id
    assert trace.path is not None and trace.path.startswith(str(tmp_path))
    events = read_events(trace.path)
    assert events[0]["event"] == "run_start"
    assert events[0]["question"] == "What is clause 15?"
    assert events[0]["retrieval_mode"] == "hybrid"
    assert events[0]["trace_id"] == trace_id


def test_step_and_final_append_events(tmp_path):
    trace = TraceLogger(trace_dir=str(tmp_path))
    trace.start("q", "vector")
    trace.step("route", {"route": "lookup"})
    trace.final([{"chunk_id": "c1"}], iterations=2, total_ms=1234)
    events = read_events(trace.path)
    assert [e["event"] for e in events] == ["run_start", "step", "final"]
    assert events[1]["step"] == "route"
    assert events[1]["detail"] == {"route": "lookup"}
    assert events[2]["citations"] == [{"chunk_id": "c1"}]
    assert events[2]["iterations"] == 2 and events[2]["total_ms"] == 1234
    assert events[1]["ts_ms"] >= 0


def test_start_with_unwritable_dir_disables_tracing(tmp_path, monkeypatch):
    # trace_dir is a *file*, so makedirs fails with an OSError
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    trace = TraceLogger(trace_dir=str(blocker))
    trace_id = trace.start("q", "hybrid")
    assert trace_id == trace.trace_id
    assert trace.path is None  # tracing silently disabled


def test_write_failure_disables_further_writes(tmp_path, monkeypatch):
    trace = TraceLogger(trace_dir=str(tmp_path))
    trace.start("q", "hybrid")

    def broken_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", broken_open)
    trace.step("route", {})  # write fails -> path cleared, exception swallowed
    assert trace.path is None
    monkeypatch.undo()
    # after the failure, step() is a no-op: no partial file, no error
    trace.final([], 0, 0)


def test_step_before_start_is_noop(tmp_path):
    trace = TraceLogger(trace_dir=str(tmp_path))
    assert trace.path is None
    trace.step("route", {})  # no path yet: must not raise
    trace.final([], 0, 0)
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_trace_ids_are_unique(tmp_path):
    one, two = TraceLogger(trace_dir=str(tmp_path)), TraceLogger(trace_dir=str(tmp_path))
    assert one.trace_id != two.trace_id