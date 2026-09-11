"""Agent container HTTP API (docs/contracts.md, agent :8001).

- GET  /health -> {"status": "ok"}
- POST /chat   -> SSE stream of step / token / done / error events
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import graph as graph_module
from app.tracing import TraceLogger

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
_log = logging.getLogger("agent")

app = FastAPI(title="graphrag-agent", version="0.1.0")

_COMPILED_GRAPH: Any = None
_GRAPH_LOCK = asyncio.Lock()


class ChatRequest(BaseModel):
    message: str
    retrieval_mode: str | None = None  # overrides RETRIEVAL_MODE env when present


def _compiled_graph() -> Any:
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        import neo4j
        from neo4j import GraphDatabase

        uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
        auth = (
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", ""),
        )
        driver: neo4j.Driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        _COMPILED_GRAPH = graph_module.build_graph(driver)
    return _COMPILED_GRAPH


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _event_stream(request: ChatRequest) -> AsyncIterator[str]:
    started = time.monotonic()
    try:
        compiled = _compiled_graph()
    except Exception as exc:  # noqa: BLE001 - surfaced as an SSE error event
        _log.exception("agent initialisation failed")
        yield _sse({"type": "error", "message": f"agent initialisation failed: {exc}"})
        return

    message = (request.message or "").strip()
    mode = (request.retrieval_mode or os.environ.get("RETRIEVAL_MODE", "hybrid")).lower()
    trace = TraceLogger()
    trace_id = trace.start(message, mode)
    if not message:
        yield _sse({"type": "error", "message": "empty message"})
        return

    _log.info("chat start mode=%s question=%r", mode, message)
    inputs = {
        "question": message,
        "retrieval_mode": mode,
        "iterations": 0,
    }
    final_state: dict[str, Any] = {}
    try:
        async for stream_mode, chunk in compiled.astream(
            inputs,
            stream_mode=["updates", "custom"],
            config={"recursion_limit": 40},
        ):
            if stream_mode == "custom":
                if isinstance(chunk, dict) and chunk.get("type") == "token":
                    yield _sse({"type": "token", "text": chunk.get("text", "")})
                continue
            for node, update in (chunk or {}).items():
                if not isinstance(update, dict):
                    continue
                final_state.update(update)
                detail, emit = graph_module.step_detail(node, update)
                if emit:
                    trace.step(node, detail)
                    yield _sse({"type": "step", "step": node, "detail": detail})
        answer = final_state.get("answer", "")
        citations = final_state.get("citations", [])
        total_ms = int((time.monotonic() - started) * 1000)
        trace.final(citations, final_state.get("iterations", 0), total_ms)
        _log.info("chat done iterations=%d citations=%d total_ms=%d",
                  final_state.get("iterations", 0), len(citations), total_ms)
        yield _sse(
            {
                "type": "done",
                "answer": answer,
                "citations": citations,
                "trace_id": trace_id,
            }
        )
    except Exception as exc:  # noqa: BLE001 - e.g. unavailable model tag
        _log.exception("chat failed")
        yield _sse({"type": "error", "message": str(exc)})