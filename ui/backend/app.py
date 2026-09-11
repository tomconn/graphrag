"""FastAPI backend for the GraphRAG PoC UI container.

Proxies the agent's SSE chat stream to the browser and serves source
documents from the read-only data/ mount. See docs/contracts.md.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

AGENT_URL = os.environ.get("AGENT_URL", "http://agent:8001")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data")).resolve()

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ui")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PoC: the browser talks only to this backend
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    retrieval_mode: Optional[str] = None  # "hybrid" | "vector" | None


def _sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Proxy the agent's POST /chat SSE stream byte-for-byte."""
    payload = {"message": req.message, "retrieval_mode": req.retrieval_mode}

    async def event_stream() -> AsyncIterator[bytes]:
        try:
            # Read timeout must exceed the agent's slowest gap between events.
            timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{AGENT_URL}/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")[:300]
                        log.warning("agent /chat returned %s: %s", resp.status_code, body)
                        yield _sse_event(
                            {"type": "error", "message": f"agent returned HTTP {resp.status_code}"}
                        )
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except (httpx.HTTPError, OSError) as exc:
            log.warning("agent /chat failed: %s", exc)
            yield _sse_event({"type": "error", "message": f"agent unreachable: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _extract_section(text: str, section: str) -> Optional[str]:
    """Return markdown from the heading at the end of `section` (a " > "-separated
    heading path) up to the next heading of equal-or-higher level."""
    parts = [p.strip() for p in section.split(">") if p.strip()]
    if not parts:
        return None

    lines = text.splitlines()
    chain: list[tuple[int, str]] = []  # current heading path as (level, title)
    target_idx: Optional[int] = None
    target_level = 0
    in_fence = False  # ignore '#' inside code fences

    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        level, title = len(m.group(1)), m.group(2).strip()
        while chain and chain[-1][0] >= level:
            chain.pop()
        chain.append((level, title))
        titles = [t for _, t in chain]
        if len(titles) >= len(parts) and titles[-len(parts):] == parts:
            target_idx, target_level = i, level
            break

    if target_idx is None:
        return None

    out = [lines[target_idx]]
    in_fence = False
    for line in lines[target_idx + 1:]:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence:
            m = HEADING_RE.match(line)
            if m and len(m.group(1)) <= target_level:
                break
        out.append(line)
    return "\n".join(out).rstrip()


@app.get("/api/source")
async def source(path: str, section: Optional[str] = None) -> JSONResponse:
    """Serve markdown (a heading section or the whole file) from DATA_DIR."""
    # Citation source_path carries the repo-relative "data/" prefix; callers
    # pass paths relative to DATA_DIR, so strip that prefix when present.
    rel = path[len("data/"):] if path.startswith("data/") else path

    # Path traversal guard: the resolved target must stay under DATA_DIR.
    target = (DATA_DIR / rel).resolve()
    if target != DATA_DIR and not target.is_relative_to(DATA_DIR):
        raise HTTPException(status_code=400, detail="path escapes DATA_DIR")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    text = target.read_text(encoding="utf-8", errors="replace")
    if section:
        content = _extract_section(text, section)
        if content is None:
            raise HTTPException(status_code=404, detail="section not found")
    else:
        content = text
    return JSONResponse({"path": path, "content": content})


@app.get("/api/health")
async def health() -> dict:
    agent = "unreachable"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{AGENT_URL}/health")
            if resp.status_code == 200:
                agent = "ok"
    except (httpx.HTTPError, OSError):
        pass
    return {"status": "ok", "agent": agent}