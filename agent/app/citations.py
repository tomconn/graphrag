"""Citation object assembly per docs/contracts.md."""

from __future__ import annotations

from typing import Any

import neo4j

_CITATION_QUERY = """
UNWIND $ids AS cid
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk {id: cid})
RETURN cid AS chunk_id,
       d.id AS doc_id, d.title AS title, d.type AS doc_type,
       d.source_path AS source_path,
       c.section AS section, c.clause AS clause, c.code_ref AS code_ref
"""


def _empty_citation() -> dict[str, str]:
    return {
        "doc_id": "",
        "title": "",
        "doc_type": "",
        "source_path": "",
        "section": "",
        "clause": "",
        "code_ref": "",
    }


def build_citations(
    driver: neo4j.Driver, chunk_ids: list[str]
) -> list[dict[str, Any]]:
    """Resolve chunk ids to citation objects (deduped, stable order)."""
    seen: set[str] = set()
    ids: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id and chunk_id not in seen and not chunk_id.startswith("graph:"):
            seen.add(chunk_id)
            ids.append(chunk_id)
    if not ids:
        return []

    def work(tx: neo4j.Transaction) -> list[neo4j.Record]:
        return list(tx.run(_CITATION_QUERY, ids=ids))

    with driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
        records = session.execute_read(work)

    citations: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for record in records:
        citation = _empty_citation()
        for field in citation:
            value = record.get(field)
            citation[field] = "" if value is None else str(value)
        key = (
            citation["doc_id"],
            citation["section"],
            citation["clause"],
            citation["code_ref"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        citations.append(citation)
    return citations