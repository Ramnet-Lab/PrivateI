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


# The relationship carries allegation_ref verbatim, exactly as the triples row
# holds it, because the reader is the one that has to decide what a tag means:
# whether the numbering starts at zero or one is inferred downstream from the
# whole set of tags, and a value normalised here would destroy the evidence that
# inference is drawn from. NULL is meaningful and is stored as such - it means
# the assertion carried no marker and is available to every allegation, not that
# it belongs to the first one.
#
# The copy on the edge is for provenance and for inspecting the graph by hand.
# It routes nothing: report generation rebuilds its own index from the triples
# table, so that table is the authority on which allegation an assertion
# answers, and a tag rewritten by a re-extraction is fresh there and stale
# here. Anyone changing routing has to change the table and the code that
# reads it; changing this property alone moves no evidence anywhere.
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
    r.allegation_ref = $allegation_ref,
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
           # sqlite3.Row indexes but has no .get(), and both of these columns
           # arrived in a migration, so a database written before it has no such
           # key at all. Asking keys() is the only way to tell a missing column
           # from a NULL one without the load dying on a KeyError.
           allegation_ref=(row["allegation_ref"]
                           if "allegation_ref" in row.keys() else None),
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

    # Said once per load rather than once per row. An absent tag column is not
    # an error - every assertion is then unrestricted, which is how the pipeline
    # behaved before the column existed - but a report built on this graph will
    # route no evidence, and without this line the reason would be invisible.
    if "allegation_ref" not in rows[0].keys():
        log.info("triples.allegation_ref is not present in this database; "
                 "assertions load without an allegation tag and stay available "
                 "to every allegation")

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
        # allegation_ref is returned so that a fact read out of the graph
        # carries the tag the edge was loaded with, which is what lets a
        # reader on the entity page see which allegation an assertion is
        # testimony about. It is not what routes evidence: report generation
        # keys off the index it rebuilds from the triples table, and that
        # table remains the authority, so a tag rewritten after a load is
        # fresh there and stale here.
        facts = [dict(r) for r in session.run(
            """MATCH (e:Entity {entity_id:$id})-[r]-(other:Entity)
               WHERE r.triple_id IS NOT NULL
               RETURN CASE WHEN startNode(r).entity_id = $id THEN 'out' ELSE 'in' END AS direction,
                      r.predicate AS predicate, other.name AS other_name,
                      other.type AS other_type, other.entity_id AS other_id,
                      r.quote AS quote, r.source_doc AS source_doc,
                      r.source_page AS source_page, r.source_file AS source_file,
                      r.event_date AS event_date,
                      r.allegation_ref AS allegation_ref
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


# An inferred edge is written with NO triple_id, and that is load-bearing rather
# than decorative. Every read path in this module - snapshot, entity_detail and
# timeline - filters on "r.triple_id IS NOT NULL", so an edge without one is
# invisible to all of them, and to every caller of them, without a single one
# having to learn that inference exists. A reader written before this feature
# cannot accidentally show a model's opinion as testimony, because it was already
# asking for testimony by name.
#
# inferred:true is for reading the graph by hand, and for the sweep below. It is
# not what provides the isolation; the absent triple_id is.
INFER = """
MATCH (a:Entity {{entity_id: $a_id}})
MATCH (b:Entity {{entity_id: $b_id}})
MERGE (a)-[r:{rel} {{link: $link_id}}]->(b)
SET r.inferred = true, r.relation = $relation, r.confidence = $confidence,
    r.basis = $basis, r.pairing = $pairing, r.model = $model,
    r.loaded_at = $now
"""


def load_links(rows: list[dict]) -> int:
    """Draw recorded entity links into the graph as inferred edges.

    Entities are MATCHed rather than MERGEd: a link names two entities that
    extraction already put in the graph, and creating one here would invent a
    node out of an inference about it.
    """
    if not rows:
        return 0
    drawn = 0
    with driver().session() as session:
        for row in rows:
            # subject_id says which way a directed relation runs; a symmetric
            # one has none, and the stored pair order is used as written.
            a_id, b_id = row["a_id"], row["b_id"]
            if row.get("subject_id") == b_id:
                a_id, b_id = b_id, a_id
            result = session.run(
                INFER.format(rel=rel_type(row["relation"])),
                a_id=a_id, b_id=b_id, link_id=f"{row['a_id']}|{row['b_id']}",
                relation=row["relation"], confidence=row.get("confidence"),
                basis=row.get("basis"), pairing=row.get("pairing"),
                model=row.get("model"), now=utcnow())
            drawn += result.consume().counters.relationships_created
    # A link naming an entity the graph does not have draws nothing, and the
    # MATCH makes that silent. It means the entity was merged or deleted after
    # the link was recorded, which is worth one line rather than a discrepancy
    # between two counts that nobody can account for later.
    if drawn < len(rows):
        log.warning("%d of %d recorded link(s) name an entity that is no longer "
                    "in the graph and were not drawn", len(rows) - drawn,
                    len(rows))
    else:
        log.info("drew %d inferred edge(s) into the graph", drawn)
    return drawn


def clear_links() -> None:
    """Remove every inferred edge, leaving the evidence graph as it was."""
    with driver().session() as session:
        session.run("MATCH ()-[r]->() WHERE r.inferred DELETE r")


def inferred_edges(limit: int = 2000) -> list[dict]:
    """Inferred edges for the graph page, asked for by name.

    A separate query rather than a widening of snapshot(): snapshot answers
    "what does the record say", and the answer to that must not change because
    a pass has been run over it.
    """
    with driver().session() as session:
        return [dict(r) for r in session.run(
            """MATCH (a:Entity)-[r]->(b:Entity) WHERE r.inferred
               RETURN a.entity_id AS source, b.entity_id AS target,
                      r.relation AS relation, r.confidence AS confidence,
                      r.basis AS basis, r.pairing AS pairing
               ORDER BY r.confidence DESC LIMIT $limit""", limit=limit)]


def merged_timeline() -> list[dict]:
    """The timeline with one entry per event, and statements kept apart.

    Two problems the raw query cannot solve, because both are about how rows
    relate to each other rather than what any row says.

    Duplicates: triple_id hashes the document and page, so two witnesses
    describing 18 March are two distinct edges and arrive as two events. Showing
    them separately overstates the record - it turns corroboration, which is
    two sources for one event, into twice as much happening. They are merged on
    what actually identifies an event (its date and the assertion made about it)
    and their citations are stacked.

    Statements: a "PERSON stated CLAIM" row is structurally identical to a fact
    row and renders identically, so "the incident with Pike was a normal
    correction delivered rudely" - a characterisation, argued about rather than
    dated - sits in the chronology as though it were a thing that happened on a
    day. They are kept, because when somebody said something is often the point,
    but marked so a reader and the report can tell the two apart.
    """
    with driver().session() as session:
        rows = [dict(r) for r in session.run(
            """MATCH (a:Entity)-[r]->(b:Entity)
               WHERE r.triple_id IS NOT NULL AND r.event_date IS NOT NULL
                     AND r.event_date <> ''
               RETURN r.event_date AS date,
                      r.event_date_basis AS basis,
                      a.name AS subject, r.predicate AS predicate,
                      b.name AS object, b.type AS object_type,
                      r.source_file AS source_file,
                      r.source_doc AS source_doc, r.source_page AS source_page,
                      r.quote AS quote
               ORDER BY r.event_date""")]

    merged: dict[tuple, dict] = {}
    for row in rows:
        predicate = (row.get("predicate") or "").strip().casefold()
        row["is_statement"] = (predicate == "stated"
                               and (row.get("object_type") or "") == "CLAIM")
        # The date is part of the identity but only to the day: the same event
        # given a time by one witness and not by another is one event.
        key = (str(row.get("date") or "")[:10],
               (row.get("subject") or "").casefold(),
               predicate,
               (row.get("object") or "").casefold())
        first = merged.get(key)
        if first is None:
            row["sources"] = [{"source_file": row.get("source_file"),
                               "source_doc": row.get("source_doc"),
                               "source_page": row.get("source_page"),
                               "quote": row.get("quote")}]
            merged[key] = row
            continue
        seen = {(s.get("source_doc"), s.get("source_page"))
                for s in first["sources"]}
        if (row.get("source_doc"), row.get("source_page")) not in seen:
            first["sources"].append({"source_file": row.get("source_file"),
                                     "source_doc": row.get("source_doc"),
                                     "source_page": row.get("source_page"),
                                     "quote": row.get("quote")})
        # Keep the more precise of two dates for the same event: a witness who
        # gave a time said more than one who gave only the day.
        if len(str(row.get("date") or "")) > len(str(first.get("date") or "")):
            first["date"] = row["date"]
            first["basis"] = row.get("basis")
    out = list(merged.values())
    out.sort(key=lambda r: (str(r.get("date") or ""), r.get("subject") or ""))
    return out


def clear() -> None:
    with driver().session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    with state.tx() as conn:
        conn.execute("UPDATE triples SET loaded_at = NULL")
