"""Generate a report against a standing investigative objective.

The objective is a list of allegations. Each one is answered on its own, from
its own evidence, because a single prompt covering everything encourages the
model to blur one allegation's evidence into another's - which is the failure
that matters most in a document like this.

Every finding must cite the file and page it rests on, and each allegation ends
in one of three dispositions:

  Substantiated       - the evidence in these documents supports it
  Not substantiated   - the evidence contradicts it
  Insufficient        - the documents do not settle it either way

"Insufficient" is a real answer and the prompt says so. A model pushed to reach
a conclusion on thin evidence will invent the missing part.
"""
from __future__ import annotations

import hashlib
import re

from . import chat, embed, graph, state
from .config import env_str
from .log import get_logger, utcnow
from .model_client import Ollama, default_options, thinking_enabled

log = get_logger("report")

OBJECTIVE_KEY = "cdi_objective"
MAX_PASSAGES = 10
MAX_RELATIONSHIPS = 40

SYSTEM = (
    "You are an investigating officer writing findings from a set of documents "
    "that have been collected and indexed. You write plainly and you do not "
    "overstate.\n"
    "\n"
    "Rules you follow without exception:\n"
    "- Every factual statement cites its source inline as [filename p.N].\n"
    "- You use only the material provided. You never rely on outside knowledge "
    "and never assume facts that the documents do not state.\n"
    "- Where the documents conflict, you say so explicitly, give both accounts "
    "with their sources, and do not silently prefer one.\n"
    "- Where the documents do not settle a question, you say the evidence is "
    "insufficient. That is a correct and expected answer.\n"
    "- You distinguish what a document records first-hand from what it reports "
    "someone else said. Second-hand accounts are identified as such.\n"
    "- When the same fact exists in a primary record (a log, a form, a system "
    "report) and in a person's restatement of it, cite the record and its "
    "custodian first; the restatement is corroboration, not the source.\n"
    "- When a witness states a limitation on their own observation - distance, "
    "an obstructed view, headphones, arriving mid-event - carry that limitation "
    "into any finding that rests on their account.\n"
    "- Allegations are substantiated or not; an objective or goal is answered, "
    "never 'substantiated'. The numbered allegations you are given are the "
    "complete list - do not invent, split, or renumber them.\n"
    "- You do not recommend discipline and you do not speculate about motive."
)

ALLEGATION_TEMPLATE = """Write the findings for this one allegation only.

ALLEGATION {number}: {allegation}

Use this structure exactly:

#### Allegation {number}: <short restatement>

**Disposition:** <Substantiated | Not substantiated | Insufficient evidence>

**Findings**

<Numbered findings. One fact each, each ending with its [filename p.N] citation.>

**Conflicts in the evidence**

<Any place the documents disagree, with both sources. Write "None identified." if there are none.>

**Gaps**

<What the documents do not establish, and what record would settle it. Write "None identified." if there are none.>

---

RELATIONSHIPS EXTRACTED FROM THE DOCUMENTS:
{relationships}

PASSAGES FROM THE DOCUMENTS:
{passages}
"""

SUMMARY_TEMPLATE = """Write the opening of an investigation report.

Use this structure exactly:

## Summary of findings

<One short paragraph per allegation stating the disposition and the single most
important reason, each with its [filename p.N] citation.>

## Persons named

<A list. For each: name, role if the documents state one, and the documents
they appear in. Note where the same person appears under more than one form of
their name.>

## Timeline

<Dated events in order, each as: DATE — event [filename p.N]. Only dates the
documents actually state.>

---

THE ALLEGATIONS AND THEIR DISPOSITIONS:
{dispositions}

DATED EVENTS EXTRACTED FROM THE DOCUMENTS:
{timeline}

PERSONS AND ORGANISATIONS EXTRACTED:
{entities}
"""


def get_objective() -> str:
    return state.get_setting(OBJECTIVE_KEY, "")


def set_objective(text: str) -> None:
    state.set_setting(OBJECTIVE_KEY, text.strip())


def split_allegations(objective: str) -> list[str]:
    """One entry per allegation.

    Accepts numbered lists, bulleted lists, or one per line - operators write
    these by hand and should not have to match a format.
    """
    text = (objective or "").strip()
    if not text:
        return []
    # Numbered items are the allegations; anything before the first number is
    # the objective's preamble - context, never Allegation 1. Treating the
    # preamble as an allegation shifts every number down and misfiles findings
    # under the wrong heading, which a graded run demonstrated in practice.
    marker = re.compile(r"(?m)^\s*(?:allegation\s*)?(?:\d+[.)]|[-*•])\s+", re.IGNORECASE)
    matches = list(marker.finditer(text))
    if matches:
        items = []
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end():end].strip()
            if body:
                items.append(body)
        if items:
            return items
    return [line.strip() for line in text.splitlines() if line.strip()] or [text]


def _format_relationships(facts: list[dict]) -> str:
    if not facts:
        return "(none found for this allegation)"
    lines = []
    for f in facts[:MAX_RELATIONSHIPS]:
        when = f" on {f['event_date']}" if f.get("event_date") else ""
        lines.append(f"- {f['subject']} {f['predicate']} {f['object']}{when} "
                     f"[{f['source_file']} p.{f['source_page']}]")
    return "\n".join(lines)


def _format_passages(passages: list[dict]) -> str:
    if not passages:
        return "(no matching passages)"
    return "\n\n".join(f"[{p['filename']} p.{p['page_num']}]\n{p['text'].strip()}"
                       for p in passages)


def _timeline_block() -> str:
    rows = graph.timeline() if graph.available() else []
    if not rows:
        return "(no dated events extracted)"
    return "\n".join(
        f"- {r['date']}: {r['subject']} {r['predicate']} {r['object']} "
        f"[{r['source_file']} p.{r['source_page']}]" for r in rows[:60])


def _entities_block() -> str:
    rows = state.query(
        """SELECT e.entity_type, e.canonical_name, e.mention_count,
                  (SELECT GROUP_CONCAT(DISTINCT d.filename)
                     FROM triples t JOIN documents d ON d.doc_id = t.doc_id
                    WHERE t.subject_name = e.canonical_name
                       OR t.object_name = e.canonical_name) AS files
           FROM entities e
           WHERE e.merged_into IS NULL AND e.entity_type IN ('PERSON','ORG')
           ORDER BY e.mention_count DESC LIMIT 40""")
    if not rows:
        return "(no entities extracted)"
    out = []
    for r in rows:
        aliases = state.query(
            "SELECT canonical_name FROM entities WHERE merged_into=?",
            (f"{r['entity_type']}:{r['canonical_name']}",))
        also = ""
        if aliases:
            also = " (also appears as " + ", ".join(a["canonical_name"] for a in aliases) + ")"
        out.append(f"- {r['entity_type']}: {r['canonical_name']}{also} — "
                   f"mentioned {r['mention_count']}x in {r['files'] or 'unknown'}")
    return "\n".join(out)


def generate(objective: str | None = None):
    """Yield ('status'|'token'|'error'|'done', payload) while writing the report."""
    objective = (objective if objective is not None else get_objective()).strip()
    if not objective:
        yield "error", "No objective has been entered."
        return

    model = env_str("TEXT_MODEL", "").strip()
    if not model:
        yield "error", "TEXT_MODEL is not set, so no report can be written."
        return

    docs = state.query_one("SELECT COUNT(*) AS n FROM documents WHERE status='done'")
    facts_total = state.query_one("SELECT COUNT(*) AS n FROM triples")
    if not facts_total or not facts_total["n"]:
        yield "error", ("No facts have been extracted yet. Upload documents and "
                        "let them finish processing first.")
        return

    client = Ollama()
    try:
        client.require_model(model, "TEXT_MODEL")
    except Exception as exc:
        yield "error", str(exc).splitlines()[0]
        return

    options = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                              "REPORT_NUM_PREDICT", 1400)
    allegations = split_allegations(objective)
    body: list[str] = []
    dispositions: list[str] = []

    for index, allegation in enumerate(allegations, 1):
        yield "status", f"Allegation {index} of {len(allegations)}"
        passages = embed.search(allegation, k=MAX_PASSAGES)
        facts = chat.relationships_for(allegation, passages)
        prompt = ALLEGATION_TEMPLATE.format(
            number=index, allegation=allegation,
            relationships=_format_relationships(facts),
            passages=_format_passages(passages))

        section = ""
        try:
            for token in client.stream(model, prompt, system=SYSTEM, options=options,
                                       think=thinking_enabled()):
                section += token
                yield "token", token
        except Exception as exc:
            log.error("allegation %d failed: %s", index, exc)
            yield "error", str(exc).splitlines()[0]
            return
        yield "token", "\n\n"
        body.append(section.strip())

        match = re.search(r"\*\*Disposition:\*\*\s*(.+)", section)
        verdict = match.group(1).strip() if match else "not stated"
        dispositions.append(f"- Allegation {index}: {allegation}\n  Disposition: {verdict}")

    yield "status", "Summary, persons, and timeline"
    prompt = SUMMARY_TEMPLATE.format(dispositions="\n".join(dispositions),
                                     timeline=_timeline_block(),
                                     entities=_entities_block())
    head = ""
    try:
        for token in client.stream(model, prompt, system=SYSTEM, options=options,
                                   think=thinking_enabled()):
            head += token
            yield "token", token
    except Exception as exc:
        yield "error", str(exc).splitlines()[0]
        return

    created = utcnow()
    full = (f"# Report of Investigation\n\n"
            f"Generated {created} from {docs['n'] if docs else 0} document(s) and "
            f"{facts_total['n']} extracted fact(s) using {model}.\n\n"
            f"## Objective\n\n{objective}\n\n"
            f"{head.strip()}\n\n## Findings by allegation\n\n"
            + "\n\n".join(body)
            + "\n\n---\n\nEvery statement above is drawn from the uploaded documents "
              "and cited to the page it came from. Machine transcription and "
              "extraction were used throughout; the page images remain the "
              "authority.\n")

    report_id = hashlib.sha256(f"{created}|{objective}".encode()).hexdigest()[:16]
    with state.tx() as conn:
        conn.execute(
            """INSERT INTO reports (report_id, objective, body, model, documents,
                                    assertions, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (report_id, objective, full, model, docs["n"] if docs else 0,
             facts_total["n"], created))
    log.info("report %s written (%d allegation(s))", report_id, len(allegations))
    yield "done", {"report_id": report_id}
