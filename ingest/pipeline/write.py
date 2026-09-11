"""Neo4j writes — exactly per docs/contracts.md ("Graph writes (ingest)").

* Idempotent Document/Chunk upserts (MERGE ... SET +=).
* Vector index chunk_embeddings + fulltext index chunk_text_ft
  (CREATE ... IF NOT EXISTS, dim = EMBEDDING_DIM).
* Entity MERGE by (label, key) — entities persist across re-ingests.
* Edges with {source_chunk, source_document} provenance; missing endpoints
  are skipped and logged.
* Re-ingest cleanup: delete the document's chunks plus ONLY edges whose
  source_document points at this document; entity nodes persist.
* Statements are batched in transactions of ~500.
"""
from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase

LOG = logging.getLogger(__name__)

BATCH_SIZE = 500

# Label/type interpolation is safe: labels/types reaching this module are
# validated against schema/graph_schema.yaml upstream (extract.py), so they
# are plain [A-Za-z]+ identifiers — Cypher cannot parameterize labels/types.
STMT_DOCUMENT = (
    "MERGE (d:Document {id: $id}) SET d += $props"
)
STMT_CHUNK = (
    "MATCH (d:Document {id: $doc_id}) "
    "MERGE (c:Chunk {id: $chunk_id}) SET c += $props "
    "MERGE (d)-[:HAS_CHUNK]->(c)"
)
STMT_ENTITY = (
    "MERGE (e:`{label}` {{key: $key}}) SET e += $props"
)
STMT_EDGE = (
    "MATCH (a:`{slabel}` {{key: $skey}}) "
    "MATCH (b:`{tlabel}` {{key: $tkey}}) "
    "MERGE (a)-[r:`{rel_type}`]->(b) "
    "SET r.source_chunk = $chunk_id, r.source_document = $doc_id"
)


class GraphWriter:
    """Batched, idempotent writes to the Neo4j GraphRAG store."""

    def __init__(self, uri: str, user: str, password: str, dim: int,
                 knowledge_labels: dict[str, str] | None = None):
        """knowledge_labels maps each knowledge-layer label to its merge-key
        property (from schema/graph_schema.yaml, e.g. {"CpsClause": "id",
        "Pattern": "name"}). Uniqueness constraints are created for each so
        concurrent MERGE writes are safe.
        """
        self.dim = int(dim)
        self.knowledge_labels = knowledge_labels or {}
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        LOG.info("Connected to Neo4j at %s", uri)
        self._ensure_indexes()
        self._ensure_constraints()

    def close(self) -> None:
        self.driver.close()

    # -- indexes -----------------------------------------------------------

    def _ensure_indexes(self) -> None:
        # EMBEDDING_DIM is validated as an int upstream; index OPTIONS do not
        # accept parameters, so the validated integer is interpolated.
        if self.dim < 1 or self.dim > 4096:
            raise ValueError(f"Unreasonable EMBEDDING_DIM: {self.dim}")
        vector_index = (
            "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
            "FOR (c:Chunk) ON c.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: " + str(self.dim) +
            ", `vector.similarity_function`: 'cosine'}}"
        )
        fulltext_index = (
            "CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS "
            "FOR (c:Chunk) ON EACH [c.text]"
        )
        with self.driver.session() as session:
            session.run(vector_index).consume()
            session.run(fulltext_index).consume()
        LOG.info("Indexes ensured: chunk_embeddings (dim=%d, cosine), "
                 "chunk_text_ft", self.dim)

    def _ensure_constraints(self) -> None:
        """Idempotent uniqueness constraints on every knowledge label's merge
        key — required for safe concurrent MERGE of entity nodes.
        """
        if not self.knowledge_labels:
            return
        # Label/property names come from the validated schema yaml (plain
        # [A-Za-z]+ identifiers); Cypher cannot parameterize either.
        with self.driver.session() as session:
            for label, key in sorted(self.knowledge_labels.items()):
                statement = (
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:`{label}`) REQUIRE n.{key} IS UNIQUE"
                )
                session.run(statement).consume()
        LOG.info("Uniqueness constraints ensured for %d labels: %s",
                 len(self.knowledge_labels),
                 ", ".join(f"{l}.{k}" for l, k in
                           sorted(self.knowledge_labels.items())))

    # -- batching ----------------------------------------------------------

    def _run_batch(self, statements: list[tuple[str, dict[str, Any]]]) -> None:
        for start in range(0, len(statements), BATCH_SIZE):
            batch = statements[start:start + BATCH_SIZE]
            with self.driver.session() as session:
                def work(tx):
                    for query, params in batch:
                        tx.run(query, params)
                session.execute_write(work)

    def _run_edges(self, edges: list[tuple[str, dict[str, Any]]]) -> None:
        """Run edge MERGEs; a statement that creates nothing means at least
        one endpoint was not found — skip and log per the contract.
        """
        for start in range(0, len(edges), BATCH_SIZE):
            batch = edges[start:start + BATCH_SIZE]
            with self.driver.session() as session:
                def work(tx):
                    for query, params in batch:
                        summary = tx.run(query, params).consume()
                        counters = summary.counters
                        if counters.relationships_created == 0 \
                                and counters.properties_set == 0:
                            LOG.warning(
                                "Skipping edge %s [%s -> %s] in doc %s: an "
                                "endpoint entity was not found in the graph",
                                params.get("rel_hint", "?"),
                                params.get("skey"), params.get("tkey"),
                                params.get("doc_id"))
                session.execute_write(work)

    # -- document / chunks -------------------------------------------------

    def replace_document(self, doc, chunks: list) -> None:
        """Idempotently upsert one document: cleanup, then Document + Chunks.

        Cleanup (contract): DETACH DELETE the document's Chunk nodes plus
        every knowledge edge with source_document = this doc; entity nodes
        persist and are re-merged below. All in one transaction.
        """
        # Contract cleanup, in null-safe form: the contract's
        # "OPTIONAL MATCH (c)<-[r]-(x) WHERE r.source_document = $id DELETE r"
        # is fully subsumed by (a) DETACH DELETE (removes every relationship
        # attached to the chunk) and (b) the sweep statement below (deletes
        # ALL edges with source_document = $id), and avoids DELETE of a null
        # r when a chunk has no provenance edges. Entity nodes persist.
        cleanup = [
            (
                "MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) "
                "DETACH DELETE c",
                {"id": doc.doc_id},
            ),
            (
                "MATCH ()-[r]->() WHERE r.source_document = $id DELETE r",
                {"id": doc.doc_id},
            ),
        ]
        with self.driver.session() as session:
            def work(tx):
                for query, params in cleanup:
                    tx.run(query, params)
            session.execute_write(work)

        statements: list[tuple[str, dict[str, Any]]] = [(
            STMT_DOCUMENT,
            {"id": doc.doc_id,
             "props": {"id": doc.doc_id, "title": doc.title,
                       "type": doc.doc_type, "source_path": doc.source_path}},
        )]
        for chunk in chunks:
            statements.append((
                STMT_CHUNK,
                {"doc_id": doc.doc_id,
                 "chunk_id": chunk.id,
                 "props": {"id": chunk.id, "text": chunk.text,
                           "embedding": chunk.embedding,
                           "section": chunk.section,
                           "clause": chunk.clause,
                           "code_ref": chunk.code_ref,
                           "type": chunk.doc_type}},
            ))
        self._run_batch(statements)
        LOG.info("[%s] upserted %d chunks", doc.doc_id, len(chunks))

    # -- knowledge layer ---------------------------------------------------

    def write_entities(self, doc_id: str,
                       entities: list) -> int:
        statements = [(
            STMT_ENTITY.format(label=entity.label),
            {"key": entity.key, "props": entity.properties},
        ) for entity in entities]
        self._run_batch(statements)
        return len(statements)

    def write_edges(self, doc_id: str, chunk_id: str,
                    relations: list) -> tuple[int, list]:
        edges = [(
            STMT_EDGE.format(slabel=rel.source_label, tlabel=rel.target_label,
                             rel_type=rel.rel_type),
            {"skey": rel.source_key, "tkey": rel.target_key,
             "chunk_id": chunk_id, "doc_id": doc_id,
             "rel_hint": rel.rel_type},
        ) for rel in relations]
        self._run_edges(edges)
        return len(edges), edges