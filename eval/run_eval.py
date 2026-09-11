#!/usr/bin/env python3
"""Scored evaluation over eval/golden_questions.jsonl against a live agent.

For every golden question the runner drives POST /chat on the agent (SSE),
in each retrieval mode (hybrid + vector), then scores:
  - citation hits:   expected (doc_title_contains, clause) matched against
                     the final citation list
  - graph paths:     expected_graph_paths verified directly against the Neo4j
                     graph (mode-independent ground truth — does the corpus
                     graph actually contain the expected path?)
  - route / iterations / total_ms from the run's own trace file

Results (including full answers — corpus-derived text) are written to
eval/results/<run_ts>/ which is git-ignored; only the printed summary and
scores are safe to commit.

Usage:  python3 eval/run_eval.py [--modes hybrid,vector] [--agent URL]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "eval" / "golden_questions.jsonl"
TRACES_DIR = REPO_ROOT / "eval" / "traces"
RESULTS_ROOT = REPO_ROOT / "eval" / "results"
DEFAULT_AGENT = os.environ.get("AGENT_URL", "http://127.0.0.1:8001")
REQUEST_TIMEOUT = 420  # generous: reasoning model + possible retries


# --------------------------------------------------------------- agent /chat

def chat(question: str, mode: str, agent_url: str) -> tuple[list[dict], dict | None]:
    """POST one question, return (events, done_event_or_error)."""
    payload = json.dumps({"message": question, "retrieval_mode": mode}).encode()
    request = urllib.request.Request(
        agent_url + "/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[dict] = []
    done: dict | None = None
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            events.append(event)
            if event.get("type") in ("done", "error"):
                done = event
                break
    return events, done


def trace_final(trace_id: str) -> dict:
    """Pull the final event (timings) from the run's trace file."""
    if not trace_id:
        return {}
    for path in TRACES_DIR.glob(f"*_{trace_id}.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "final":
                return event
    return {}


# --------------------------------------------------------------- graph paths

def _norm(name: str) -> str:
    """Lowercase, strip non-alphanumerics: 'Payment Order Processing
    Platform' and 'payment_order_processing_platform' must match."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _matches(expected: str, actual: str) -> bool:
    """Loose node-name match: containment either way after normalization
    (golden paths use descriptive names, the graph uses real names)."""
    e, a = _norm(expected), _norm(actual)
    return bool(e) and bool(a) and (e in a or a in e)


def load_graph(client: str) -> tuple[list[dict], list[dict]]:
    """All node names/labels and directed relationship triples, via
    cypher-shell in the neo4j container (reads NEO4J_PASSWORD from .env)."""
    password = ""
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        if line.startswith("NEO4J_PASSWORD="):
            password = line.split("=", 1)[1].strip()
    cypher = (
        "MATCH (n) RETURN labels(n)[0] AS label, "
        "coalesce(n.name, n.id, n.path) AS name, id(n) AS id "
        "UNION MATCH (a)-[r]->(b) RETURN "
        "type(r) AS label, "
        "coalesce(a.name, a.id, a.path) + ' -[' + type(r) + ']-> ' + "
        "coalesce(b.name, b.id, b.path) AS name, id(r) AS id"
    )
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "neo4j", "cypher-shell",
         "-u", "neo4j", "-p", password, cypher],
        capture_output=True, text=True, check=True, cwd=REPO_ROOT,
    )
    nodes: list[dict] = []
    edges: list[dict] = []
    for line in result.stdout.splitlines():
        parts = line.split(", ")
        if len(parts) != 3:
            continue
        label, name, _id = parts
        if "->" in name:  # edge row
            match = re.match(r"(.+) -\[(.+)\]-> (.+)", name)
            if match:
                edges.append({"source": match.group(1), "type": match.group(2),
                              "target": match.group(3)})
        else:
            nodes.append({"label": label, "name": name})
    del client
    return nodes, edges


def path_exists(path: list[str], nodes: list[dict], edges: list[dict]) -> tuple[bool, str]:
    """Expected graph path present? Alternates node names and rel types.
    Single-element paths check node existence only. Returns (ok, near_miss)."""
    def node_hit(expected: str) -> bool:
        return any(_matches(expected, n["name"]) for n in nodes)

    def edge_hit(expected_node: str, rel: str, next_node: str) -> bool:
        for e in edges:
            if rel.upper() != e["type"].upper():
                continue
            if _matches(expected_node, e["source"]) and _matches(next_node, e["target"]):
                return True
        return False

    if len(path) == 1:
        if node_hit(path[0]):
            return True, ""
        return False, f"node {path[0]!r} not found in graph"

    if edge_hit(path[0], path[1], path[2]):
        return True, ""
    # near-miss diagnostics: which end is loose?
    src_ok = node_hit(path[0])
    tgt_ok = node_hit(path[2])
    if not src_ok and not tgt_ok:
        return False, f"neither {path[0]!r} nor {path[2]!r} found as nodes"
    rel_types = sorted({e["type"] for e in edges})
    # near-miss: are the endpoints connected by a differently-typed edge?
    alt = sorted({e["type"] for e in edges
                  if _matches(path[0], e["source"]) and _matches(path[2], e["target"])})
    if alt:
        return False, f"endpoints connected via {', '.join(alt)} instead of {path[1]!r}"
    if path[1].upper() not in [r.upper() for r in rel_types]:
        return False, f"edge type {path[1]!r} does not exist in the graph"
    return False, f"nodes exist ({path[0]!r}={src_ok}, {path[2]!r}={tgt_ok}) but no {path[1]!r} edge"


# ------------------------------------------------------------------- scoring

def score_citations(expected: list[dict], citations: list[dict]) -> tuple[list[dict], int]:
    """Each expectation matches a citation whose title contains the phrase
    (and, when given, whose clause equals the expected clause)."""
    results = []
    hits = 0
    for exp in expected:
        phrase = exp.get("doc_title_contains", "")
        clause = exp.get("clause")
        matched = None
        for cit in citations:
            if phrase.lower() in (cit.get("title") or "").lower():
                if clause is None or str(cit.get("clause") or "") == str(clause):
                    matched = cit
                    break
        results.append({"expected": exp, "hit": matched is not None,
                        "matched_section": (matched or {}).get("section", "")})
        hits += 1 if matched is not None else 0
    return results, hits


# --------------------------------------------------------------------- main

def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="hybrid,vector")
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    golden = [json.loads(l) for l in GOLDEN_PATH.read_text().splitlines() if l.strip()]
    nodes, edges = load_graph("cypher-shell")
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for gq in golden:
        path_checks = [dict(path=p) | dict(zip(("ok", "near_miss"), path_exists(p, nodes, edges)))
                       for p in gq.get("expected_graph_paths", [])]
        for mode in modes:
            started = time.monotonic()
            events, done = chat(gq["question"], mode, args.agent)
            wall = round(time.monotonic() - started, 1)
            trace_id = (done or {}).get("trace_id", "")
            final = trace_final(trace_id)
            citations = (done or {}).get("citations", []) if done else []
            cit_results, cit_hits = score_citations(gq.get("expected_citations", []), citations)
            route = next((e["detail"].get("route") for e in events
                          if e.get("type") == "step" and e.get("step") == "route"), None)
            row = {
                "id": gq["id"], "mode": mode, "route": route,
                "wall_s": wall, "total_ms": final.get("total_ms"),
                "iterations": final.get("iterations"),
                "error": (done or {}).get("message") if (done or {}).get("type") == "error" else None,
                "citation_hits": f"{cit_hits}/{len(gq.get('expected_citations', []))}",
                "citation_detail": cit_results,
                "n_citations": len(citations),
                "graph_paths": path_checks,
                "answer": (done or {}).get("answer", "") if done else "",
                "trace_id": trace_id,
            }
            rows.append(row)
            flags = f"cit {row['citation_hits']}"
            status = "ERR" if row["error"] else "ok"
            print(f"{gq['id']:4s} {mode:6s} {status} {wall:6.1f}s route={route} {flags}",
                  flush=True)
            (out_dir / f"{gq['id']}_{mode}.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False))
            time.sleep(2)  # gentle pacing for the shared cloud endpoint

    # summary
    print("\n=== summary ===")
    header = f"{'id':4s} {'mode':6s} {'cit':5s} {'paths':>5s} {'route':18s} {'iter':4s} {'total_s':>7s}"
    print(header)
    for row in rows:
        paths_ok = sum(1 for p in row["graph_paths"] if p["ok"])
        paths_total = len(row["graph_paths"])
        print(f"{row['id']:4s} {row['mode']:6s} {row['citation_hits']:5s} "
              f"{str(paths_ok) + '/' + str(paths_total):>5s} "
              f"{str(row['route']):18s} {str(row['iterations']):4s} "
              f"{(row['total_ms'] or 0) / 1000:7.1f}"
              + ("  ERROR: " + row["error"] if row["error"] else ""))

    (out_dir / "summary.json").write_text(
        json.dumps({"run_ts": run_ts, "rows": [
            {k: v for k, v in r.items() if k != "answer"} | {"answer_chars": len(r["answer"])}
            for r in rows]}, indent=2, ensure_ascii=False))
    print(f"\nresults: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(run())