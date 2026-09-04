"""Answering questions about the documents.

Two kinds of context go to the model, because they answer different questions:

  passages       - the wording of the pages themselves, found by embedding
                   similarity.  Good for "what did the memo say about X".
  relationships  - the graph's edges around whatever the question is about.
                   Good for "who else was at Building 220", which no single
                   passage states because the answer is spread over documents.

The model is told to answer only from what it is given and to cite the file and
page, so an answer can always be walked back to a page image.
"""
from __future__ import annotations


from . import embed, graph, state
from .config import env_str
from .entities import normalize
from .log import get_logger
from .model_client import Ollama, default_options, thinking_enabled

log = get_logger("chat")

MAX_RELATIONSHIPS = 45
MAX_PASSAGES = 8
MAX_HISTORY_TURNS = 6

SYSTEM = (
    "You answer questions about a collection of documents that have already been "
    "read and indexed. You are given passages from those documents and "
    "relationships extracted from them.\n"
    "\n"
    "Rules:\n"
    "- Answer only from the passages and relationships provided. They are the "
    "whole of what is known.\n"
    "- Cite the source of each claim inline as [filename p.N], using the labels "
    "given with the material.\n"
    "- If the material does not answer the question, say so plainly and say what "
    "it does cover. Do not speculate and do not use outside knowledge.\n"
    "- When the material conflicts, say that it conflicts and give both sides "
    "with their sources.\n"
    "- Be concise and concrete. Quote the document's own wording where the exact "
    "phrasing matters."
)


def _entities_in(text: str) -> list[dict]:
    """Entities whose name appears in the given text.

    Matched on the normalised form, so "SSgt Smith" in a question finds the
    entity stored as "Smith".
    """
    haystack = f" {normalize(text)} "
    found = []
    for row in state.query(
            "SELECT entity_id, entity_type, canonical_name FROM entities "
            "WHERE merged_into IS NULL"):
        name = normalize(row["canonical_name"])
        if len(name) < 3:
            continue
        if f" {name} " in haystack:
            found.append({"entity_id": row["entity_id"], "name": row["canonical_name"],
                          "type": row["entity_type"]})
    return found


def relationships_for(question: str, passages: list[dict]) -> list[dict]:
    """Graph edges around whatever the question and the passages are about."""
    if not graph.available():
        return []

    seen: dict[str, dict] = {}
    for item in _entities_in(question):
        seen[item["entity_id"]] = item
    if len(seen) < 6:
        for passage in passages[:4]:
            for item in _entities_in(passage["text"]):
                seen.setdefault(item["entity_id"], item)

    facts: list[dict] = []
    known: set[str] = set()
    for entity in list(seen.values())[:10]:
        detail = graph.entity_detail(entity["entity_id"])
        for fact in detail.get("facts", []):
            key = (f"{detail['name']}|{fact['predicate']}|{fact['other_name']}"
                   f"|{fact['source_doc']}|{fact['source_page']}")
            if key in known:
                continue
            known.add(key)
            subject, obj = ((detail["name"], fact["other_name"])
                            if fact["direction"] == "out"
                            else (fact["other_name"], detail["name"]))
            facts.append({
                "subject": subject, "predicate": fact["predicate"], "object": obj,
                "event_date": fact.get("event_date"),
                "source_file": fact.get("source_file") or fact.get("source_doc"),
                "source_doc": fact.get("source_doc"),
                "source_page": fact.get("source_page"),
                "quote": fact.get("quote"),
            })
            if len(facts) >= MAX_RELATIONSHIPS:
                return facts
    return facts


def build_prompt(question: str, passages: list[dict], facts: list[dict],
                 history: list[dict]) -> str:
    blocks: list[str] = []

    if facts:
        lines = ["RELATIONSHIPS EXTRACTED FROM THE DOCUMENTS:"]
        for f in facts:
            when = f" on {f['event_date']}" if f.get("event_date") else ""
            lines.append(f"- {f['subject']} {f['predicate']} {f['object']}{when} "
                         f"[{f['source_file']} p.{f['source_page']}]")
        blocks.append("\n".join(lines))

    if passages:
        lines = ["PASSAGES FROM THE DOCUMENTS:"]
        for p in passages:
            lines.append(f"\n[{p['filename']} p.{p['page_num']}]\n{p['text'].strip()}")
        blocks.append("\n".join(lines))

    if not blocks:
        blocks.append("No documents have been indexed yet.")

    if history:
        turns = []
        for turn in history[-MAX_HISTORY_TURNS:]:
            role = "Question" if turn.get("role") == "user" else "Answer"
            turns.append(f"{role}: {turn.get('content', '').strip()}")
        blocks.append("EARLIER IN THIS CONVERSATION:\n" + "\n".join(turns))

    blocks.append(f"QUESTION: {question.strip()}")
    return "\n\n".join(blocks)


def sources_of(passages: list[dict], facts: list[dict]) -> list[dict]:
    """One entry per page actually put in front of the model."""
    seen: dict[tuple, dict] = {}
    for p in passages:
        seen.setdefault((p["doc_id"], p["page_num"]),
                        {"doc_id": p["doc_id"], "page_num": p["page_num"],
                         "filename": p["filename"], "from": "passage"})
    for f in facts:
        if not f.get("source_doc"):
            continue
        seen.setdefault((f["source_doc"], f["source_page"]),
                        {"doc_id": f["source_doc"], "page_num": f["source_page"],
                         "filename": f["source_file"], "from": "relationship"})
    return list(seen.values())


def answer(question: str, history: list[dict] | None = None):
    """Yield (kind, payload) as the answer is produced.

    kind is 'sources', 'token', 'error' or 'done'.
    """
    model = env_str("TEXT_MODEL", "").strip()
    if not model:
        yield "error", ("No chat model is set. Put a model name in TEXT_MODEL "
                        "in .env and restart.")
        return

    try:
        passages = embed.search(question, k=MAX_PASSAGES)
    except Exception as exc:
        log.error("retrieval failed: %s", exc)
        yield "error", str(exc).splitlines()[0]
        return

    facts = relationships_for(question, passages)
    yield "sources", {"pages": sources_of(passages, facts),
                      "relationships": len(facts), "passages": len(passages)}

    prompt = build_prompt(question, passages, facts, history or [])
    options = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                              "CHAT_NUM_PREDICT", 900)
    client = Ollama()
    try:
        client.require_model(model, "TEXT_MODEL")
        for token in client.stream(model, prompt, system=SYSTEM, options=options,
                                   think=thinking_enabled()):
            yield "token", token
    except Exception as exc:
        log.error("chat generation failed: %s", exc)
        yield "error", str(exc).splitlines()[0]
        return
    yield "done", None
