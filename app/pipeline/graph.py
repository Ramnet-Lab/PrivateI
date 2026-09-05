"""Neo4j: load assertions, and answer the questions the graph page asks.

Every relationship carries the source document, the page, and the quote it came
from, so any edge on screen can show what it was drawn from.
"""
from __future__ import annotations

import re

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

from . import state
from .config import env_str
from .entities import display_name, entity_id, resolve_canonical
from .log import get_logger, utcnow

log = get_logger("graph")

ENTITY_LABELS = {"PERSON", "ORG", "LOCATION", "EVENT", "DOCUMENT", "CLAIM"}
_REL_SAFE = re.compile(r"[^A-Z0-9_]")

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
]

_driver = None


def driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            env_str("NEO4J_URI", "bolt://neo4j:7687"),
            auth=(env_str("NEO4J_USER", "neo4j"), env_str("NEO4J_PASSWORD", "")))
    return _driver


def available() -> bool:
    try:
        driver().verify_connectivity()
        return True
    except (ServiceUnavailable, AuthError, Neo4jError, ValueError):
        return False


def rel_type(predicate: str) -> str:
    """Relationship types cannot be parameterised in Cypher, so this string is
    interpolated into the query - which makes sanitising it a security control,
    not a tidiness one."""
    rel = _REL_SAFE.sub("_", predicate.strip().upper().replace(" ", "_")).strip("_")
    rel = re.sub(r"_+", "_", rel) or "RELATED_TO"
    if rel[0].isdigit():
        rel = f"R_{rel}"
    return rel[:60]


LOAD = """
MERGE (s:Entity {{entity_id: $subject_id}})
  ON CREATE SET s.created_at = $now
SET s.name = $subject_name, s.type = $subject_type
SET s:{subject_label}
MERGE (o:Entity {{entity_id: $object_id}})
  ON CREATE SET o.created_at = $now
SET o.name = $object_name, o.type = $object_type
SET o:{object_label}
MERGE (s)-[r:{rel} {{triple_id: $triple_id}}]->(o)
SET r.predicate = $predicate, r.source_doc = $doc_id, r.source_page = $page_num,
    r.quote = $quote, r.event_date = $event_date,
    r.event_date_basis = $event_date_basis, r.source_file = $filename,
    r.source_kind = $source_kind, r.source_role = $source_role,
    r.loaded_at = $now
"""


def _load_one(tx, row, subject_id, object_id, subject_name, object_name, filename,
              source_kind="unknown", source_role=""):
    query = LOAD.format(
        subject_label=row["subject_type"] if row["subject_type"] in ENTITY_LABELS else "ENTITY",
        object_label=row["object_type"] if row["object_type"] in ENTITY_LABELS else "ENTITY",
        rel=rel_type(row["predicate"]))
    tx.run(query, subject_id=subject_id, object_id=object_id,
           subject_name=subject_name, object_name=object_name,
           subject_type=row["subject_type"], object_type=row["object_type"],
           predicate=row["predicate"], triple_id=row["triple_id"],
           doc_id=row["doc_id"], page_num=row["page_num"], quote=row["quote"],
           event_date=row["event_date"],
           event_date_basis=(row["event_date_basis"]
                             if "event_date_basis" in row.keys() else None),
           filename=filename, source_kind=source_kind,
           source_role=source_role, now=utcnow())


def load(doc_id: str | None = None, on_progress=lambda _m: None) -> int:
    sql = "SELECT * FROM triples WHERE loaded_at IS NULL"
    params: list = []
    if doc_id:
        sql += " AND doc_id = ?"
        params.append(doc_id)
    rows = state.query(sql, params)
    if not rows:
        return 0

    # dict(r), not the Row itself: sqlite3.Row supports indexing but has no
    # .get(), and the miss only shows up at load time.
    docs = {r["doc_id"]: dict(r) for r in
            state.query("SELECT doc_id, filename, doc_kind, doc_role FROM documents")}
    names = {k: v["filename"] for k, v in docs.items()}

    loaded = 0
    with driver().session() as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)
        for idx, row in enumerate(rows, 1):
            if idx % 25 == 0:
                on_progress(f"loading assertion {idx}/{len(rows)}")
            subject_id = resolve_canonical(entity_id(row["subject_type"], row["subject_name"]))
            object_id = resolve_canonical(entity_id(row["object_type"], row["object_name"]))
            session.execute_write(_load_one, row, subject_id, object_id,
                                  display_name(subject_id), display_name(object_id),
                                  names.get(row["doc_id"], row["doc_id"]),
                                  (docs.get(row["doc_id"]) or {}).get("doc_kind") or "unknown",
                                  (docs.get(row["doc_id"]) or {}).get("doc_role") or "")
            with state.tx() as conn:
                conn.execute("UPDATE triples SET loaded_at=? WHERE triple_id=?",
                             (utcnow(), row["triple_id"]))
            loaded += 1
    log.info("loaded %d assertion(s) into the graph", loaded)
    return loaded


def snapshot(limit: int = 1500) -> dict:
    """Nodes and edges for the graph page."""
    with driver().session() as session:
        nodes = [dict(r) for r in session.run(
            """MATCH (e:Entity)
               OPTIONAL MATCH (e)-[r]-() WHERE r.triple_id IS NOT NULL
               RETURN e.entity_id AS id, e.name AS name, e.type AS type,
                      count(r) AS degree
               ORDER BY degree DESC LIMIT $limit""", limit=limit)]
        keep = {n["id"] for n in nodes}
        edges = [dict(r) for r in session.run(
            """MATCH (a:Entity)-[r]->(b:Entity) WHERE r.triple_id IS NOT NULL
               RETURN a.entity_id AS source, b.entity_id AS target,
                      r.predicate AS predicate, r.source_file AS source_file,
                      r.source_doc AS source_doc, r.source_page AS source_page,
                      r.quote AS quote, r.event_date AS event_date,
                      r.triple_id AS triple_id
               LIMIT $limit""", limit=limit * 4)]
    edges = [e for e in edges if e["source"] in keep and e["target"] in keep]
    return {"nodes": nodes, "edges": edges}


def entity_detail(eid: str) -> dict:
    with driver().session() as session:
        node = session.run(
            "MATCH (e:Entity {entity_id:$id}) RETURN e.name AS name, e.type AS type",
            id=eid).single()
        if not node:
            return {}
        facts = [dict(r) for r in session.run(
            """MATCH (e:Entity {entity_id:$id})-[r]-(other:Entity)
               WHERE r.triple_id IS NOT NULL
               RETURN CASE WHEN startNode(r).entity_id = $id THEN 'out' ELSE 'in' END AS direction,
                      r.predicate AS predicate, other.name AS other_name,
                      other.type AS other_type, other.entity_id AS other_id,
                      r.quote AS quote, r.source_doc AS source_doc,
                      r.source_page AS source_page, r.source_file AS source_file,
                      r.event_date AS event_date
               ORDER BY r.event_date, r.predicate""", id=eid)]
    return {"id": eid, "name": node["name"], "type": node["type"], "facts": facts}


def timeline() -> list[dict]:
    with driver().session() as session:
        return [dict(r) for r in session.run(
            """MATCH (a:Entity)-[r]->(b:Entity)
               WHERE r.triple_id IS NOT NULL AND r.event_date IS NOT NULL
                     AND r.event_date <> ''
               RETURN r.event_date AS date,
                      r.event_date_basis AS basis,
                      a.name AS subject, r.predicate AS predicate,
                      b.name AS object, r.source_file AS source_file,
                      r.source_doc AS source_doc, r.source_page AS source_page,
                      r.quote AS quote
               ORDER BY r.event_date""")]


def clear() -> None:
    with driver().session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    with state.tx() as conn:
        conn.execute("UPDATE triples SET loaded_at = NULL")
