"""Text2Cypher per docs/contracts.md.

- Prompt: question + the ``knowledge_layer`` section of SCHEMA_FILE (labels,
  relationship triples with descriptions, provenance note) + previous error
  when retrying.
- Guardrail: any statement containing a write keyword is rejected.
- Execution: EXPLAIN first, then run, in a read Neo4j session. Max 3 attempts;
  on final failure the caller falls back to hybrid retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

import neo4j
import yaml

_log = logging.getLogger(__name__)

_DEFAULT_SCHEMA_FILE = "/app/schema/graph_schema.yaml"

# Contract: reject statements containing these keywords (case-insensitive),
# scanned outside string literals so text like `WHERE n.note = 'please CALL back'`
# is not a false positive. The mask replaces literal contents (escaped quotes
# included) with empty strings, so keywords inside values are ignored.
# LOAD and FOREACH are rejected beyond the bare write-keyword list: LOAD CSV
# FROM '<url>' is legal in a read-only session but makes the agent issue
# outbound requests (SSRF), and FOREACH nests write clauses. URL literals are
# rejected outright for the same reason — no generated query ever needs one.
_STRING_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|DETACH|DROP|REMOVE|CALL|LOAD|FOREACH)\b",
    re.IGNORECASE,
)
_URL_LITERAL_RE = re.compile(r"https?://", re.IGNORECASE)
MAX_ATTEMPTS = 3
MAX_ROWS = 25
MAX_ROW_TEXT_CHARS = 600
# Neo4j rejects a RETURN with two columns of the same name (a common LLM slip
# when an alias and the provenance note collide, e.g. two `AS implementer`
# columns). Duplicate explicit aliases are renamed deterministically instead
# of spending a retry on it.
_NO_RETRY_ERRORS = ("empty Cypher statement",)
MAX_PROMPT_ERROR_CHARS = 400


def load_schema(schema_file: str | None = None) -> dict[str, Any]:
    path = schema_file or os.environ.get("SCHEMA_FILE", _DEFAULT_SCHEMA_FILE)
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def render_knowledge_layer(schema: dict[str, Any]) -> str:
    """Render node labels + relationship triples with descriptions."""
    knowledge = schema.get("knowledge_layer", {})
    lines: list[str] = ["NODE LABELS:"]
    for label, definition in knowledge.get("node_labels", {}).items():
        props = ", ".join(definition.get("properties", []))
        key = definition.get("key", "")
        description = definition.get("description", "")
        lines.append(f"- :{label} (key: {key}; properties: {props}) — {description}")
    lines.append("")
    lines.append("RELATIONSHIPS:")
    for rel_type, definition in knowledge.get("relationships", {}).items():
        description = definition.get("description", "")
        for source in definition.get("source", []):
            for target in definition.get("target", []):
                lines.append(
                    f"- (: {source})-[:{rel_type}]->(: {target}) — {description}"
                )
    provenance = knowledge.get("edge_provenance", {})
    if provenance:
        lines.append("")
        lines.append(
            "PROVENANCE: every relationship carries "
            + ", ".join(f"`{name}` ({typ})" for name, typ in provenance.items())
            + ". Include relationship provenance in RETURN "
            "(e.g. RETURN type(r) AS rel_type, r.source_chunk AS source_chunk, "
            "r.source_document AS source_document, properties of a and b) so "
            "answers can cite the source chunk."
        )
    return "\n".join(lines)


def _build_prompt(
    question: str,
    knowledge_layer: str,
    previous_cypher: str | None = None,
    previous_error: str | None = None,
) -> str:
    prompt = (
        "You are an expert Neo4j Cypher developer. Given the graph schema and a "
        "question, write ONE Cypher query that answers the question.\n\n"
        "GRAPH SCHEMA (knowledge layer):\n"
        f"{knowledge_layer}\n\n"
        "RULES:\n"
        "- READ-ONLY query only: never use CREATE, MERGE, DELETE, SET, DETACH, "
        "DROP, REMOVE or CALL.\n"
        "- Every RETURN column must have a unique alias (AS ...); duplicate "
        "column names are an error.\n"
        "- Return at most 25 rows (add a LIMIT).\n"
        "- Use only the node labels and relationship types above.\n"
        "- A relationship type connects ONLY the endpoint pairs listed above. "
        "If no listed pair links two labels directly (e.g. System to "
        "Obligation), do not invent an edge — the association must be reached "
        "by traversing intermediate nodes that ARE connected.\n"
        "- Output ONLY the Cypher query, no explanation, no code fences.\n\n"
        f"QUESTION: {question}\n"
    )
    if previous_cypher and previous_error:
        prompt += (
            "\nPREVIOUS ATTEMPT FAILED.\n"
            f"Previous statement:\n{previous_cypher}\n"
            f"Error:\n{previous_error}\n"
            "Fix the statement and output only the corrected Cypher.\n"
        )
    return prompt


def _strip_code_fences(text: str) -> str:
    match = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _mask_string_literals(statement: str) -> str:
    """Same-length mask: string-literal contents become spaces (quotes too), so
    structural analysis — keyword positions, top-level commas, paren depth —
    never trips over quoted text, and positions still index the original."""
    out = list(statement)
    quote: str | None = None
    i = 0
    while i < len(statement):
        ch = statement[i]
        if quote:
            out[i] = " "
            if ch == "\\" and i + 1 < len(statement):
                out[i + 1] = " "
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        i += 1
    return "".join(out)


_ALIAS_RE = re.compile(r"\bAS\s+(\w+)(\s*)$", re.IGNORECASE)
# Ends the RETURN projection. The (?<!\.) keeps property access like
# `n.limit` / `n.skip` from matching.
_PROJECTION_END_RE = re.compile(
    r"(?<!\.)\b(ORDER\s+BY|LIMIT|SKIP)\b", re.IGNORECASE
)


def _dedupe_return_aliases(statement: str) -> str:
    """Rename duplicate explicit RETURN aliases so column names are unique.

    Neo4j rejects duplicate result columns; the provenance guidance makes
    collisions likely (an alias like `implementer` next to a second
    `r.x AS implementer`). Only the LAST top-level RETURN clause is scanned
    and only later duplicates are renamed (`name` -> `name_2`), so a column
    downstream code reads by name keeps its name.
    """
    masked = _mask_string_literals(statement)
    return_matches = list(re.finditer(r"\bRETURN\b", masked, re.IGNORECASE))
    if not return_matches:
        return statement
    clause_start = return_matches[-1].start()
    end = _PROJECTION_END_RE.search(masked, clause_start)
    end = end.start() if end else len(statement)
    clause_masked = masked[clause_start:end]

    # Item boundaries: commas at paren depth 0 (paren/commas inside literals
    # are already masked out).
    items: list[tuple[int, int]] = []
    item_start = 0
    depth = 0
    for idx, ch in enumerate(clause_masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append((item_start, idx))
            item_start = idx + 1
    items.append((item_start, len(clause_masked)))

    seen: set[str] = set()
    replacements: list[tuple[int, int, str]] = []
    for lo, hi in items:
        item = clause_masked[lo:hi]
        match = _ALIAS_RE.search(item)
        if not match:
            continue
        alias = match.group(1)
        if alias not in seen:
            seen.add(alias)
            continue
        counter = 2
        while f"{alias}_{counter}" in seen:
            counter += 1
        new_alias = f"{alias}_{counter}"
        seen.add(new_alias)
        # Keep the trailing whitespace (often the newline before ORDER BY).
        replacement = f"AS {new_alias}{match.group(2)}"
        replacements.append((clause_start + lo + match.start(), clause_start + hi, replacement))

    if not replacements:
        return statement
    out = statement
    for start, end, text in reversed(replacements):
        out = out[:start] + text + out[end:]
    return out


def _clean_statement(cypher: str) -> str:
    """Strip fences and a trailing semicolon; reject write keywords
    (string-literal contents are excluded from the keyword scan) and URL
    literals (scanned on the unmasked statement — in valid Cypher a URL can
    only appear inside a string literal, so masking would hide it).
    """
    statement = _strip_code_fences(cypher).strip().rstrip(";").strip()
    if not statement:
        raise ValueError("empty Cypher statement")
    if _URL_LITERAL_RE.search(statement):
        raise ValueError("statement contains a URL literal; not permitted")
    masked = _STRING_LITERAL_RE.sub("''", statement)
    if _WRITE_KEYWORDS.search(masked):
        raise ValueError("statement contains a write keyword; read-only Cypher only")
    return _dedupe_return_aliases(statement)


def _validate_and_run(
    driver: neo4j.Driver, statement: str
) -> tuple[list[dict[str, Any]], str]:
    """EXPLAIN (validation) then run, in a read session. Returns (rows, error)."""

    def work(tx: neo4j.Transaction, sql_statement: str):
        tx.run("EXPLAIN " + sql_statement).consume()  # validation pass
        return [record.data() for record in tx.run(sql_statement)]

    try:
        with driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
            rows = session.execute_read(
                lambda tx: work(tx, statement)
            )
        return rows[:MAX_ROWS], ""
    except Exception as exc:  # noqa: BLE001 - driver errors are retried
        return [], str(exc)


def _collect_chunk_refs(rows: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            candidates: list[str] = []
            if key == "source_chunk" and isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, dict) and isinstance(
                value.get("source_chunk"), str
            ):
                candidates.append(value["source_chunk"])
            for ref in candidates:
                if ref and ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
    return refs


def rows_to_text(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows[:MAX_ROWS]:
        line = json.dumps(row, ensure_ascii=False, default=str)
        lines.append(line[:MAX_ROW_TEXT_CHARS])
    return "\n".join(lines)


def _trim(text: str, limit: int = MAX_PROMPT_ERROR_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " [truncated]"


def run_text2cypher(
    driver: neo4j.Driver,
    question: str,
    schema: dict[str, Any],
    complete_fn: Callable[..., str],
) -> dict[str, Any] | None:
    """Generate, guard and execute a Cypher query. Returns
    ``{"cypher": str, "rows": list, "chunk_refs": list}`` or None on failure."""
    knowledge_layer = render_knowledge_layer(schema)
    previous_cypher: str | None = None
    previous_error: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = _build_prompt(question, knowledge_layer, previous_cypher, previous_error)
        raw = complete_fn(prompt)
        try:
            statement = _clean_statement(raw)
        except ValueError as exc:
            message = str(exc)
            if message in _NO_RETRY_ERRORS:
                message = (
                    "The previous response contained no Cypher statement. "
                    "Respond with ONLY the Cypher query, no other text."
                )
            previous_cypher, previous_error = _trim(raw) or "(no output)", message
            _log.warning("text2cypher attempt %d rejected: %s", attempt, exc)
            continue
        rows, error = _validate_and_run(driver, statement)
        if error:
            previous_cypher, previous_error = statement, _trim(error)
            _log.warning("text2cypher attempt %d failed: %s", attempt, error)
            continue
        _log.info("text2cypher succeeded after %d attempt(s)", attempt)
        return {
            "cypher": statement,
            "rows": rows,
            "chunk_refs": _collect_chunk_refs(rows),
        }

    _log.warning("text2cypher exhausted %d attempts; falling back", MAX_ATTEMPTS)
    return None