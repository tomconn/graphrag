"""JSONL tracing per docs/contracts.md trace format.

One file per run at ``$TRACE_DIR/<utc_ts>_<trace_id>.jsonl``. Write failures
are swallowed: tracing must never break a chat.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

_log = logging.getLogger(__name__)

_DEFAULT_TRACE_DIR = "/app/eval/traces"


class TraceLogger:
    def __init__(self, trace_dir: str | None = None) -> None:
        self.trace_dir = trace_dir or os.environ.get("TRACE_DIR", _DEFAULT_TRACE_DIR)
        self.trace_id = uuid.uuid4().hex
        self.path: str | None = None
        self._started_at = time.monotonic()
        self._file_written = False

    def start(self, question: str, retrieval_mode: str) -> str:
        """Create the trace dir, record run_start, return the trace id."""
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            utc_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            self.path = os.path.join(self.trace_dir, f"{utc_ts}_{self.trace_id}.jsonl")
            self._write(
                {
                    "event": "run_start",
                    "trace_id": self.trace_id,
                    "retrieval_mode": retrieval_mode,
                    "question": question,
                }
            )
        except OSError:
            _log.warning("trace dir unavailable at %s; tracing disabled", self.trace_dir)
            self.path = None
        return self.trace_id

    def step(self, step: str, detail: dict[str, Any]) -> None:
        self._write(
            {
                "event": "step",
                "step": step,
                "detail": detail,
                "ts_ms": int((time.monotonic() - self._started_at) * 1000),
            }
        )

    def final(
        self, citations: list[dict[str, Any]], iterations: int, total_ms: int
    ) -> None:
        self._write(
            {
                "event": "final",
                "citations": citations,
                "iterations": iterations,
                "total_ms": total_ms,
            }
        )

    def _write(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception:  # noqa: BLE001 - tracing must never break a chat
            _log.debug("trace write failed", exc_info=True)
            self.path = None