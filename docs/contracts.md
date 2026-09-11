# Shared contracts

Implementation contracts shared across the three services. The README is the design contract; this file pins the wire-level details.

## Ports and environment

| Service | Port | Reads |
|---|---|---|
| FastAPI backend (ui container) | `8000` (published `${FASTAPI_PORT}`) | `AGENT_URL` (default `http://agent:8001`), `DATA_DIR` (mounted `data/`, read-only) |
| Agent (LangGraph) | `8001` (published `${AGENT_PORT}`) | `NEO4J_*`, `OLLAMA_*`, `EMBEDDING_MODEL/DIM`, `RETRIEVAL_MODE`, `SCHEMA_FILE` (`/app/schema/graph_schema.yaml`), `TRACE_DIR` (`/app/eval/traces`) |
| Ingest | one-off | `NEO4J_*`, `OLLAMA_*`, `EMBEDDING_MODEL/DIM`, `CHUNK_SIZE/CHUNK_OVERLAP`, `SCHEMA_FILE` |
| Neo4j | 7474 / 7687 | `NEO4J_PASSWORD` |

Python base images: `python:3.11-slim`. The agent and ingest each get `./schema:/app/schema:ro`; agent also gets `./eval:/app/eval`; ingest gets `./data:/app/data:ro`; ui gets `./data:/app/data:ro`.

## Agent HTTP API (agent container, :8001)

### `GET /health` → `200 {"status": "ok"}` (used by compose-era checks and the UI backend)

### `POST /chat` → SSE stream (`text/event-stream`)

Request:
```json
{"message": "user question", "retrieval_mode": "hybrid | vector | null"}
```
`retrieval_mode` overrides the `RETRIEVAL_MODE` env when present (evaluation toggle).

Response: `text/event-stream`; each event is a line `data: <json>\n\n` with one of these shapes:

```json
{"type": "step", "step": "route|rewrite|retrieve|traverse|synthesize", "detail": {...}}
{"type": "token", "text": "..."}                                  // synthesis tokens, in order
{"type": "done", "answer": "full text", "citations": [...], "trace_id": "..."}
{"type": "error", "message": "..."}
```

`step` events are advisory progress for the UI's retrieval panel. `detail` is free-form
(e.g. `{"route": "relationship"}`, `{"rewritten": "..."}`, `{"chunks": 8}`,
`{"cypher": "MATCH ..."}`). The stream ends after `done` or `error`.

### Citation object (shared shape, used in `done.citations` and UI rendering)

```json
{
  "doc_id": "cps-234-information-security",
  "title": "CPS 234 — Information Security (SYNTHETIC EXTRACT)",
  "doc_type": "regulatory",
  "source_path": "data/regulatory/cps-234-information-security.md",
  "section": "Physical security of information assets > Clause 27",
  "clause": "27",                     // regulatory only, "" otherwise
  "code_ref": ""                      // code only, "path#Symbol[:lines]" otherwise
}
```

## UI backend HTTP API (ui container, :8000)

- `POST /api/chat` — body/semantics identical to agent `/chat`; proxies the SSE stream
  byte-for-byte (httpx `stream`). Adds CORS for browser use.
- `GET /api/source?path=<relative path under data/>&section=<heading path>` →
  `{"path": "...", "content": "<markdown of the requested section or whole file>"}`.
  **Path traversal guard: resolve under `DATA_DIR`, reject `..`.** Used by citation
  links ("View source") in the UI.
- `GET /api/health` → `200 {"status": "ok", "agent": "ok|unreachable"}`.

## Trace format (agent writes to `$TRACE_DIR/<utc_ts>_<trace_id>.jsonl`)

One JSON object per line, one file per run:
```json
{"event": "run_start", "trace_id": "...", "retrieval_mode": "hybrid", "question": "..."}
{"event": "step", "step": "route", "detail": {...}, "ts_ms": 123}
{"event": "step", "step": "retrieve", "detail": {"chunk_ids": [...], "scores": [...]}, "ts_ms": 456}
{"event": "final", "citations": [...], "iterations": 2, "total_ms": 8900}
```

## Graph writes (ingest) — exact Cypher contract

- Upsert: `MERGE (d:Document {id: $id}) SET d += $props`
- Chunks: `MERGE (c:Chunk {id: $chunk_id}) SET c += $props MERGE (d)-[:HAS_CHUNK]->(c)`
- Indexes (idempotent `CREATE INDEX ... IF NOT EXISTS`):
  - `CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS FOR (c:Chunk) ON c.embedding OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}`
  - `CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]`
- Entity merge: `MERGE (e:<Label> {key: $normalized_name_or_id}) SET e += $props`
- Edges: `MERGE (a)-[r:<TYPE>]->(b) SET r.source_chunk = $chunk_id, r.source_document = $doc_id`
  (entities looked up by merge key; skip + log if an endpoint was not found)
- Re-ingest cleanup (per document, in one transaction):
  ```cypher
  MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk)
  OPTIONAL MATCH (c)<-[r]-(x) WHERE r.source_document = $id DELETE r
  DETACH DELETE c
  // then delete edges whose source_document = $id regardless of chunk linkage
  MATCH ()-[r]->() WHERE r.source_document = $id DELETE r
  // entity nodes persist; re-merge on write
  ```
  Note: delete edges with `r.source_document = $id` **only** — edges extracted from
  other documents survive.

## Text2Cypher contract (agent)

- Prompt: question + `knowledge_layer` section of `SCHEMA_FILE` (labels, relationship
  triples with descriptions, provenance note) + previous error if retrying.
- Execution: Neo4j session in **read** mode; `EXPLAIN` the statement first (validation),
  then run. Retry with the driver error appended, max 3 attempts; on final failure fall
  back to hybrid retrieval.
- Any generated statement containing `CREATE|MERGE|DELETE|SET|DETACH|DROP|REMOVE|CALL`
  (case-insensitive) is rejected before execution.

## Coding conventions

- Python: stdlib + minimal deps; every container has `requirements.txt` pinned loosely
  (`package>=x,<major+1`). Type hints on public functions. `os.environ` config only.
- Logging: `logging` module, INFO default.
- No tests required beyond module importability; `python -m compileall` must pass.