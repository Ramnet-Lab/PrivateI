"""Comparing entities with each other, which nothing else in the pipeline does.

Extraction reads one page at a time.  It is given a page and asked what that
page asserts, so every assertion it produces is true of that page alone and
related to nothing outside it.  That is the right shape for extraction - an
assertion that reached across documents could not cite the page it came from -
but it means the corpus is never compared with itself.  What Rao described in
March and what Whitcomb described in June arrive as two unconnected assertions
however plainly they are the same event, and the graph shows it: of 203 CLAIM
nodes in the first corpus this was written against, 200 had exactly one edge.

This pass compares them.  One pass per category pairing, twenty-one of them,
because "are these two related" is a different question for two claims than for
a person and a document, and one prompt asked of all of them returns answers
general enough to be useless.  Each pairing brings its own vocabulary.

Everything here is inference.  Not one row it writes came out of a document;
each is a model's reading of two things that did.  The whole design follows from
that: a link that cannot say in the entities' own words why it exists is not
recorded, "unrelated" is a real answer rather than silence, and the graph edges
written from these rows carry no triple_id - which is what keeps them out of
every evidence path that predates them.
"""
from __future__ import annotations

import itertools
import json
import re
from typing import Any, Iterable, Iterator

import numpy as np

from . import embed, entities, graph, llm_settings, state
from .config import env_float, env_int, env_str
from .log import get_logger, utcnow
from .model_client import Ollama, default_options, random_seed, thinking_enabled

log = get_logger("links")

TYPES = ("CLAIM", "DOCUMENT", "EVENT", "LOCATION", "ORG", "PERSON")

# How many pairs go into one model call.  This is the whole reason an exhaustive
# run is possible at all: every pair is still put to the model and judged on its
# own merits, but 33,000 pairs become 1,300 calls instead of 33,000, which is
# the difference between a night and a fortnight.  Coverage does not change -
# only how many pairs share a prompt.  Too large and the model starts answering
# about the batch instead of about each pair, which is what the per-line format
# and the fallback below are for.
BATCH_PAIRS = env_int("LINK_BATCH_PAIRS", 25)

# Cosine similarity at which two entities are worth comparing in connected mode.
# Only used to SELECT pairs, never to decide a relation - the model does that.
SIMILARITY = env_float("LINK_SIMILARITY", 0.82)

# Below this the model's own stated confidence is treated as no link found. A
# link nobody is confident about is noise in a graph whose purpose is to make
# the non-obvious visible.
MIN_CONFIDENCE = env_float("LINK_MIN_CONFIDENCE", 0.5)

# How many entities go on each side of a grouped window.  A window of two
# blocks shows 2*WINDOW entities and covers WINDOW*WINDOW pairs, at the same
# prompt size as WINDOW pairs shown one by one - because a pair costs two entity
# texts either way, and a window stops repeating them.  That is a 25-fold
# reduction in calls for identical coverage, which is what makes comparing
# everything against everything a thing that finishes.
WINDOW = env_int("LINK_WINDOW", 25)

NONE = ""       # the relation of a pair that was judged and found unrelated

# The model writes this on its own line when it has finished a window. Without
# it an empty reply and a reply meaning "none of these connect" are the same
# string, and treating a failure as "none connect" would mark every pair in the
# window judged and never look at them again.
TERMINATOR = "END OF ANSWERS"


class Pairing:
    """One category pairing, its relations, and how to ask about them."""

    __slots__ = ("a", "b", "relations", "directed", "question")

    def __init__(self, a: str, b: str, question: str,
                 relations: dict[str, bool]) -> None:
        # Types are held in sorted order so a pairing has one name however it
        # is looked up, which is also the order the stored pair key uses.
        self.a, self.b = (a, b) if a <= b else (b, a)
        self.question = question
        self.relations = list(relations)
        # Which relations have a direction. "reports to" needs to know which of
        # the two reports to the other; "same person" does not, and asking would
        # invite an answer to a question with no meaning.
        self.directed = relations

    @property
    def name(self) -> str:
        return f"{self.a}:{self.b}"

    def __repr__(self) -> str:            # for logs and test failures
        return f"<Pairing {self.name}>"


def _pairings() -> dict[str, Pairing]:
    """Every pairing of every type, each with a vocabulary of its own.

    The vocabularies are deliberately short.  A model given fifteen relations to
    choose between spreads its answers across all fifteen; given four that
    actually differ, it picks the one that fits or says none of them do.
    """
    spec: list[Pairing] = [
        Pairing("CLAIM", "CLAIM",
                "Both are statements a witness made. Decide whether they are "
                "about the same thing, and if so how they stand to each other.",
                {"corroborates": False, "contradicts": False,
                 "elaborates": True, "restates": False}),
        Pairing("CLAIM", "DOCUMENT",
                "Decide whether the document is where the statement was made "
                "or recorded.",
                {"recorded in": True}),
        Pairing("CLAIM", "EVENT",
                "Decide whether the statement is about that event.",
                {"describes": True, "dates": True}),
        Pairing("CLAIM", "LOCATION",
                "Decide whether the statement places something at that place.",
                {"places at": True}),
        Pairing("CLAIM", "ORG",
                "Decide whether the statement is about that organisation.",
                {"concerns": True}),
        Pairing("CLAIM", "PERSON",
                "Decide whether the statement is about that person, or was "
                "made by them.",
                {"is about": True, "attributed to": True}),
        Pairing("DOCUMENT", "DOCUMENT",
                "Decide whether these are the same document, or one refers to "
                "or replaces the other.",
                {"same document": False, "refers to": True,
                 "supersedes": True}),
        Pairing("DOCUMENT", "EVENT",
                "Decide whether the document records that event.",
                {"records": True}),
        Pairing("DOCUMENT", "LOCATION",
                "Decide whether the document concerns that place.",
                {"mentions": True}),
        Pairing("DOCUMENT", "ORG",
                "Decide whether the organisation issued the document or is its "
                "subject.",
                {"issued by": True, "mentions": True}),
        Pairing("DOCUMENT", "PERSON",
                "Decide whether the person wrote, signed, or is the subject of "
                "the document.",
                {"authored by": True, "is an interview of": True,
                 "mentions": True}),
        Pairing("EVENT", "EVENT",
                "Decide whether these are one occurrence described twice, or "
                "whether one led to the other.",
                {"same occurrence": False, "preceded": True, "caused": True}),
        Pairing("EVENT", "LOCATION",
                "Decide whether the event happened at that place.",
                {"happened at": True}),
        Pairing("EVENT", "ORG",
                "Decide whether the organisation was involved in the event.",
                {"involved": True}),
        Pairing("EVENT", "PERSON",
                "Decide whether the person was present at or party to the "
                "event.",
                {"was present at": True, "was involved in": True}),
        Pairing("LOCATION", "LOCATION",
                "Decide whether these are the same place, or one is inside the "
                "other.",
                {"same place": False, "contains": True}),
        Pairing("LOCATION", "ORG",
                "Decide whether the organisation is at that place.",
                {"located at": True}),
        Pairing("LOCATION", "PERSON",
                "Decide whether the person was at that place.",
                {"was at": True}),
        Pairing("ORG", "ORG",
                "Decide whether these are the same organisation, or one is "
                "part of the other.",
                {"same organisation": False, "part of": True}),
        Pairing("ORG", "PERSON",
                "Decide whether the person belongs to or leads the "
                "organisation.",
                {"member of": True, "leads": True}),
        Pairing("PERSON", "PERSON",
                "Decide whether these name the same person, or how the two "
                "stand to one another.",
                {"same person": False, "reports to": True,
                 "related to": False, "adverse to": False}),
    ]
    table = {p.name: p for p in spec}
    # Every combination is covered or the pass would silently have a blind spot
    # the operator was never told about.
    expected = {f"{a}:{b}" for a, b in
                itertools.combinations_with_replacement(TYPES, 2)}
    missing = expected - set(table)
    if missing:                              # a programming error, not input
        raise RuntimeError(f"no pairing defined for {sorted(missing)}")
    return table


PAIRINGS = _pairings()


# -- what counts as a claim -------------------------------------------------------

# A CLAIM should be a proposition - something that can be corroborated or
# contradicted. In practice it is whatever the extractor could not place:
# validate() re-types every unrecognised label to CLAIM, and the extraction
# prompt positively instructs that a job title arrive as one. So the graph fills
# with "1500", "March", "cables", "section NCOIC" - names of things, not
# assertions about them. Comparing those is comparing noise, so they are left
# out of pair selection and counted out loud.
_VERBISH = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"said|stated|told|saw|heard|asked|went|came|gave|took|made|"
    r"would|could|should|will|shall|may|might|must|can|"
    r"\w+ed|\w+ing|\w+s)\b", re.IGNORECASE)


def is_proposition(text: str) -> bool:
    """Whether a CLAIM value is an assertion rather than the name of a thing.

    Deliberately crude and deliberately generous: this decides only what is
    compared, never what is true, and excluding a real claim costs more than
    including a fragment. Four words and something verb-shaped is the whole
    test.
    """
    words = text.split()
    if len(words) < 4:
        return False
    return bool(_VERBISH.search(text))


# -- the corpus as entities -------------------------------------------------------

def _canonical(entity_type: str, name: str) -> str:
    return entities.resolve_canonical(entities.entity_id(entity_type, name))


def corpus_entities() -> dict[str, list[dict]]:
    """Every distinct entity, by type, with where in the corpus it appears.

    Read from triples rather than the entities table because the provenance -
    which document, which page, which date - is what connected mode selects on,
    and the entities table does not carry it.
    """
    rows = state.query(
        """SELECT subject_type AS t, subject_name AS n, doc_id, page_num,
                  event_date FROM triples
           UNION ALL
           SELECT object_type, object_name, doc_id, page_num, event_date
           FROM triples""")
    seen: dict[str, dict] = {}
    for row in rows:
        etype = row["t"]
        if etype not in TYPES:
            continue
        eid = _canonical(etype, row["n"])
        item = seen.get(eid)
        if item is None:
            item = seen[eid] = {"id": eid, "type": etype,
                                "name": entities.display_name(eid) or row["n"],
                                "docs": set(), "pages": set(), "dates": set()}
        item["docs"].add(row["doc_id"])
        item["pages"].add((row["doc_id"], row["page_num"]))
        if row["event_date"]:
            # Day precision. Two things on the same day are worth comparing;
            # two things at the same minute are the same thing said twice, and
            # the pass should not need a clock to notice it.
            item["dates"].add(str(row["event_date"])[:10])
    by_type: dict[str, list[dict]] = {t: [] for t in TYPES}
    for item in seen.values():
        by_type[item["type"]].append(item)
    for group in by_type.values():
        group.sort(key=lambda e: e["id"])
    return by_type


def _usable(by_type: dict[str, list[dict]], keep_fragments: bool) -> tuple[
        dict[str, list[dict]], dict[str, int]]:
    """The entities a run will actually compare, and what it left out."""
    if keep_fragments:
        return by_type, {}
    kept = dict(by_type)
    claims = by_type.get("CLAIM", [])
    usable = [c for c in claims if is_proposition(c["name"])]
    kept["CLAIM"] = usable
    skipped = len(claims) - len(usable)
    return kept, ({"CLAIM": skipped} if skipped else {})


# -- choosing pairs ---------------------------------------------------------------

def _connected(a: dict, b: dict, near: set[tuple[str, str]]) -> bool:
    """Whether two entities have any reason to be compared.

    Sharing a page, a document or a day is a reason; so is being close in
    embedding space. None of these decides the relation - they decide only
    whether the model is asked.
    """
    if a["pages"] & b["pages"] or a["docs"] & b["docs"]:
        return True
    if a["dates"] & b["dates"]:
        return True
    return (a["id"], b["id"]) in near


def _near_pairs(group_a: list[dict], group_b: list[dict],
                same: bool) -> set[tuple[str, str]]:
    """Pairs whose names embed close together, for connected mode only.

    Best effort. No embedding model configured is not an error here - it costs
    the run one selection signal out of four, and saying so once is better than
    refusing to run at all.
    """
    model = env_str("EMBED_MODEL", "")
    if not model:
        return set()
    texts = [e["name"] for e in group_a] + ([] if same else
                                            [e["name"] for e in group_b])
    if len(texts) < 2:
        return set()
    try:
        client = embed._client()
        client.require_model(model, "EMBED_MODEL")
        vectors = np.asarray(client.embed(model, texts), dtype=np.float32)
    except Exception as exc:
        log.warning("no similarity signal for %s: %s",
                    group_a[0]["type"] if group_a else "?", exc)
        return set()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    scores = vectors @ vectors.T
    near: set[tuple[str, str]] = set()
    split = len(group_a)
    for i, first in enumerate(group_a):
        js = range(i + 1, split) if same else range(split, len(texts))
        for j in js:
            if scores[i, j] >= SIMILARITY:
                other = group_a[j] if same else group_b[j - split]
                near.add(tuple(sorted((first["id"], other["id"]))))
    return near


def pairs_for(pairing: Pairing, by_type: dict[str, list[dict]],
              mode: str) -> list[tuple[dict, dict]]:
    """The pairs this pairing will put to the model, in this mode."""
    same = pairing.a == pairing.b
    group_a = by_type.get(pairing.a, [])
    group_b = group_a if same else by_type.get(pairing.b, [])
    if not group_a or not group_b:
        return []
    raw = (itertools.combinations(group_a, 2) if same
           else itertools.product(group_a, group_b))
    if mode in ("exhaustive", "grouped"):
        return list(raw)
    near = _near_pairs(group_a, group_b, same)
    return [(a, b) for a, b in raw if _connected(a, b, near)]


def estimate(mode: str = "exhaustive", keep_fragments: bool = False) -> dict:
    """How much a run would cost, counted from the corpus actually loaded.

    Computed rather than asserted, and shown before anything starts, because an
    exhaustive run over a corpus this size is hours and the operator is the one
    who gets to decide whether to spend them.
    """
    by_type = corpus_entities()
    usable, skipped = _usable(by_type, keep_fragments)
    done = judged_pairs()
    rows = []
    total = outstanding = calls = 0
    for name, pairing in sorted(PAIRINGS.items()):
        pairs = pairs_for(pairing, usable, mode)
        left = sum(1 for a, b in pairs
                   if tuple(sorted((a["id"], b["id"]))) not in done)
        total += len(pairs)
        outstanding += left
        if mode == "grouped":
            # One call per window that still has an unjudged pair in it.
            same = pairing.a == pairing.b
            group_a = usable.get(pairing.a, [])
            group_b = group_a if same else usable.get(pairing.b, [])
            for wl, wr in windows(group_a, group_b, same):
                if any(tuple(sorted((a["id"], b["id"]))) not in done
                       for a, b in window_pairs(wl, wr, same)):
                    calls += 1
        else:
            calls += (left + BATCH_PAIRS - 1) // BATCH_PAIRS
        rows.append({"pairing": name, "pairs": len(pairs), "outstanding": left,
                     "relations": pairing.relations})
    return {
        "mode": mode,
        "pairings": rows,
        "pairs": total,
        "outstanding": outstanding,
        "calls": calls,
        "already_judged": total - outstanding,
        "entities": {t: len(v) for t, v in usable.items()},
        "skipped": skipped,
        "batch": WINDOW if mode == "grouped" else BATCH_PAIRS,
    }


# -- asking ------------------------------------------------------------------------

SYSTEM = (
    "You compare pairs of items drawn from an investigation's records and say "
    "whether each pair is connected. You are not deciding what happened. You "
    "are deciding whether two items refer to the same thing or stand in a "
    "stated relation to each other.\n"
    "Most pairs are not connected. NONE is the expected answer and there is no "
    "cost to giving it; a connection asserted between two unrelated items is "
    "worse than a connection missed, because it will be read as a finding.\n"
    "Never infer a connection from the fact that two items appear in the same "
    "investigation. Everything here appears in the same investigation."
)

TEMPLATE = """{question}

Answer for every numbered pair, one line each, in this exact format:

<number>. <relation> | <subject> | <confidence> | <basis>

- <relation> is exactly one of: {relations}, or NONE if the two are not connected.
- <subject> is A or B, naming which side the relation runs FROM, for these
  relations: {directed}. For any other relation, and for NONE, write a dash.
- <confidence> is a number between 0 and 1.
- <basis> is one short sentence saying what in the two items themselves shows
  the connection. If you cannot write that sentence from the items as given,
  the answer is NONE.

Write nothing else - no preamble, no summary, no blank lines between answers.

PAIRS

{pairs}
"""


def _render(pairs: list[tuple[dict, dict]]) -> str:
    out = []
    for n, (a, b) in enumerate(pairs, 1):
        out.append(f"{n}.\n"
                   f"   A ({a['type']}): {a['name']}\n"
                   f"   B ({b['type']}): {b['name']}")
    return "\n".join(out)


_ANSWER = re.compile(r"^\s*(\d+)\s*[.)]\s*(.+)$")


def parse_answer(raw: str, pairing: Pairing, count: int) -> dict[int, dict]:
    """Read one answer per pair out of the reply, keyed by pair number.

    Lenient about everything except which pair an answer belongs to. A verdict
    attached to the wrong pair is a false link between two entities that were
    never compared, which is the one error here that would be invisible
    afterwards - so a line whose number is missing or out of range is dropped
    rather than guessed at.
    """
    known = {r.casefold(): r for r in pairing.relations}
    found: dict[int, dict] = {}
    for line in raw.splitlines():
        match = _ANSWER.match(line)
        if not match:
            continue
        number = int(match.group(1))
        if not 1 <= number <= count or number in found:
            continue
        fields = [f.strip() for f in match.group(2).split("|")]
        relation = fields[0].strip(" *`").casefold() if fields else "none"
        if relation in ("none", "-", "") or relation not in known:
            found[number] = {"relation": NONE}
            continue
        subject = fields[1].strip().upper()[:1] if len(fields) > 1 else ""
        try:
            confidence = float(re.sub(r"[^\d.]", "", fields[2]) or 0)
        except (ValueError, IndexError):
            confidence = 0.0
        basis = fields[3].strip() if len(fields) > 3 else ""
        found[number] = {
            "relation": known[relation],
            "subject": subject if subject in ("A", "B") else "",
            "confidence": max(0.0, min(1.0, confidence)),
            "basis": basis,
        }
    return found


def _verdicts(client, model: str, pairing: Pairing,
              pairs: list[tuple[dict, dict]], options: dict) -> dict[int, dict]:
    directed = [r for r in pairing.relations if pairing.directed[r]] or ["none"]
    prompt = TEMPLATE.format(
        question=pairing.question,
        relations=", ".join(pairing.relations),
        directed=", ".join(directed),
        pairs=_render(pairs))
    answer = client.generate(model, prompt, system=SYSTEM, options=options,
                             think=thinking_enabled())
    return parse_answer((answer.get("response") or ""), pairing, len(pairs))


def judge(client, model: str, pairing: Pairing,
          pairs: list[tuple[dict, dict]], options: dict) -> list[dict]:
    """Judge a batch, falling back to single pairs if the batch cannot be read.

    A batch answer that is missing pairs is not treated as those pairs being
    unrelated. Silence is not an answer, and recording it as one would write
    "judged, nothing there" against pairs the model never addressed - which,
    because a judged pair is never asked again, would bury them permanently.
    """
    verdicts = _verdicts(client, model, pairing, pairs, options)
    missing = [n for n in range(1, len(pairs) + 1) if n not in verdicts]
    if missing and len(pairs) > 1:
        log.warning("%s: %d of %d pair(s) unanswered in a batch; asking singly",
                    pairing.name, len(missing), len(pairs))
        for n in missing:
            try:
                one = _verdicts(client, model, pairing, [pairs[n - 1]], options)
            except Exception as exc:
                log.warning("%s: pair %d failed on its own too: %s",
                            pairing.name, n, exc)
                continue
            if 1 in one:
                verdicts[n] = one[1]
    out = []
    for n, (a, b) in enumerate(pairs, 1):
        verdict = verdicts.get(n)
        if verdict is None:
            continue                     # never answered; left for a later run
        out.append({"a": a, "b": b, **verdict})
    return out


# -- asking about many at once ------------------------------------------------------

def blocks(items: list[dict], width: int) -> list[list[dict]]:
    return [items[i:i + width] for i in range(0, len(items), width)]


def windows(group_a: list[dict], group_b: list[dict], same: bool,
            width: int = WINDOW) -> list[tuple[list[dict], list[dict]]]:
    """Tile the pair space with windows of entities, covering every pair once.

    Cut each side into blocks and take every combination of blocks. A pair of
    entities falls in exactly one window - the window made of the block holding
    one and the block holding the other - so showing every window to the model
    shows it every pair, and shows none of them twice.

    This is what makes comparing everything against everything affordable. Asked
    pair by pair, a batch of 25 pairs renders 50 entity texts to cover 25 pairs.
    A window of two 25-blocks renders the same 50 texts and covers 625. The
    saving is not a shortcut past any pair; it is the removal of the repetition
    in writing each entity out once per pair it appears in.
    """
    if same:
        parts = blocks(group_a, width)
        return [(parts[i], parts[j])
                for i in range(len(parts)) for j in range(i, len(parts))]
    return [(a, b) for a in blocks(group_a, width)
            for b in blocks(group_b, width)]


def window_pairs(left: list[dict], right: list[dict],
                 same: bool) -> list[tuple[dict, dict]]:
    """Every pair a window covers, including the block-against-itself case."""
    if left is right:
        return list(itertools.combinations(left, 2))
    if same:
        # Two different blocks of one type: every cross pair, and no self pairs
        # because an entity is only ever in one block.
        return [(a, b) for a in left for b in right]
    return [(a, b) for a in left for b in right]


GROUP_TEMPLATE = """{question}

Below are two sets of items. Consider every possible pairing of one item from
SET A with one item from SET B{selfnote}. Most pairs are not connected; report
only the ones that are.

For each connected pair write one line:

<A-label> | <B-label> | <relation> | <subject> | <confidence> | <basis>

- the labels are exactly as written below, for example A3 and B7.
- <relation> is exactly one of: {relations}
- <subject> is A or B, naming which side the relation runs FROM, for these
  relations: {directed}. For any other relation, write a dash.
- <confidence> is a number between 0 and 1.
- <basis> is one short sentence saying what in the two items themselves shows
  the connection. If you cannot write that sentence, do not report the pair.

Report nothing for pairs that are not connected. When you have written every
connected pair - and if there are none, immediately - write this on its own
line and stop:

{terminator}

SET A

{set_a}

SET B

{set_b}
"""

_GROUP_ANSWER = re.compile(
    r"^\s*([AB])\s*(\d+)\s*[|,]\s*([AB])\s*(\d+)\s*\|(.*)$", re.IGNORECASE)


def parse_group(raw: str, pairing: Pairing, left: list[dict],
                right: list[dict]) -> tuple[list[dict], bool]:
    """Read named pairs out of a window answer.

    Returns the connections found and whether the model actually finished. The
    second half matters more than the first: a window is only marked judged when
    the model said it was done, because marking it on a truncated or failed
    reply would bury every pair in it - hundreds at a time - as "looked at,
    nothing there".
    """
    complete = TERMINATOR.casefold() in raw.casefold()
    known = {r.casefold(): r for r in pairing.relations}
    sides = {"A": left, "B": right}
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for line in raw.splitlines():
        match = _GROUP_ANSWER.match(line.strip().strip("*`-"))
        if not match:
            continue
        a_side, a_num, b_side, b_num, rest = match.groups()
        a_side, b_side = a_side.upper(), b_side.upper()
        try:
            a = sides[a_side][int(a_num) - 1]
            b = sides[b_side][int(b_num) - 1]
        except (KeyError, IndexError, ValueError):
            continue                      # a label for an item not in this window
        if a["id"] == b["id"]:
            continue                      # an entity is not connected to itself
        key = tuple(sorted((a["id"], b["id"])))
        if key in seen:
            continue
        fields = [f.strip() for f in rest.split("|")]
        relation = fields[0].strip(" *`").casefold() if fields else ""
        if relation not in known:
            continue
        subject = fields[1].strip().upper()[:1] if len(fields) > 1 else ""
        try:
            confidence = float(re.sub(r"[^\d.]", "", fields[2]) or 0)
        except (ValueError, IndexError):
            confidence = 0.0
        basis = fields[3].strip() if len(fields) > 3 else ""
        seen.add(key)
        out.append({"a": a, "b": b, "relation": known[relation],
                    "subject": subject if subject in ("A", "B") else "",
                    "confidence": max(0.0, min(1.0, confidence)),
                    "basis": basis})
    return out, complete


def _label(items: list[dict], side: str) -> str:
    return "\n".join(f"{side}{i}. {e['name']}" for i, e in enumerate(items, 1))


def judge_window(client, model: str, pairing: Pairing, left: list[dict],
                 right: list[dict], options: dict) -> tuple[list[dict], bool]:
    directed = [r for r in pairing.relations if pairing.directed[r]] or ["none"]
    same_block = left is right
    prompt = GROUP_TEMPLATE.format(
        question=pairing.question,
        relations=", ".join(pairing.relations),
        directed=", ".join(directed),
        terminator=TERMINATOR,
        selfnote=(". SET A and SET B are the same set, so consider every pair "
                  "of two different items from it" if same_block else ""),
        set_a=_label(left, "A"),
        set_b=("(the same as SET A)" if same_block else _label(right, "B")))
    answer = client.generate(model, prompt, system=SYSTEM, options=options,
                             think=thinking_enabled())
    return parse_group((answer.get("response") or ""), pairing, left,
                       right if not same_block else left)


# -- storing -----------------------------------------------------------------------

def judged_pairs() -> set[tuple[str, str]]:
    """Every pair already judged, so a resumed run does not ask them again."""
    return {(r["a_id"], r["b_id"])
            for r in state.query("SELECT a_id, b_id FROM entity_links")}


def record(verdict: dict, pairing: Pairing, model: str, run_id: str) -> bool:
    """Write one judgement. True when it recorded an actual link."""
    a, b = verdict["a"], verdict["b"]
    a_id, b_id = sorted((a["id"], b["id"]))
    relation = verdict.get("relation") or NONE
    basis = (verdict.get("basis") or "").strip()
    confidence = float(verdict.get("confidence") or 0.0)

    # A link with no basis, or one nobody is confident of, is recorded as no
    # link rather than dropped: it has been judged, and the point of storing the
    # judgement is that it is never asked again.
    if relation and (not basis or confidence < MIN_CONFIDENCE):
        log.info("%s: %s / %s discarded (%s, confidence %.2f, basis %s)",
                 pairing.name, a["name"][:40], b["name"][:40], relation,
                 confidence, "present" if basis else "missing")
        relation = NONE

    subject_id = None
    if relation and pairing.directed.get(relation):
        side = verdict.get("subject")
        subject_id = a["id"] if side == "A" else b["id"] if side == "B" else None
    with state.tx() as conn:
        conn.execute(
            """INSERT INTO entity_links (a_id, b_id, pairing, relation,
                                         subject_id, confidence, basis, model,
                                         run_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(a_id, b_id) DO UPDATE SET
                   relation=excluded.relation, subject_id=excluded.subject_id,
                   confidence=excluded.confidence, basis=excluded.basis,
                   model=excluded.model, run_id=excluded.run_id,
                   created_at=excluded.created_at""",
            (a_id, b_id, pairing.name, relation, subject_id,
             confidence if relation else None, basis if relation else None,
             model, run_id, utcnow()))
    return bool(relation)


def found(pairing: str | None = None, limit: int = 500) -> list[dict]:
    """Recorded links, strongest first, for the page that shows them."""
    sql = ["SELECT * FROM entity_links WHERE relation <> ''"]
    params: list = []
    if pairing:
        sql.append("AND pairing = ?")
        params.append(pairing)
    sql.append("ORDER BY confidence DESC, pairing, a_id LIMIT ?")
    params.append(limit)
    rows = [dict(r) for r in state.query(" ".join(sql), params)]
    for row in rows:
        row["a_name"] = entities.display_name(row["a_id"])
        row["b_name"] = entities.display_name(row["b_id"])
    return rows


def summary() -> dict:
    """Counts for the stat line: how many judged, how many were links."""
    row = state.query_one(
        "SELECT COUNT(*) AS judged, "
        "SUM(CASE WHEN relation <> '' THEN 1 ELSE 0 END) AS links "
        "FROM entity_links")
    by_pairing = [dict(r) for r in state.query(
        "SELECT pairing, COUNT(*) AS links FROM entity_links "
        "WHERE relation <> '' GROUP BY pairing ORDER BY links DESC")]
    return {"judged": int(row["judged"] or 0) if row else 0,
            "links": int(row["links"] or 0) if row else 0,
            "by_pairing": by_pairing}


def draw() -> int:
    """Redraw every recorded link into the graph as an inferred edge.

    The stored rows are the authority and the graph is a projection of them, so
    this wipes the inferred edges and lays them down again rather than trying to
    reconcile two sets. There are hundreds of these, not millions.
    """
    if not graph.available():
        log.warning("no graph database, so nothing was drawn")
        return 0
    graph.clear_links()
    rows = [dict(r) for r in state.query(
        "SELECT * FROM entity_links WHERE relation <> ''")]
    return graph.load_links(rows)


def clear() -> int:
    """Throw the whole inference set away.

    Called whenever the corpus changes. Inferences are derived from the corpus
    they were drawn over, so a document arriving or leaving makes every one of
    them suspect - and working out which ones is harder and less trustworthy
    than computing them again. Declaring the lot stale is the honest move.
    """
    row = state.query_one("SELECT COUNT(*) AS n FROM entity_links")
    count = int(row["n"]) if row else 0
    with state.tx() as conn:
        conn.execute("DELETE FROM entity_links")
    # The graph copy goes with the rows it was drawn from. Left behind it would
    # be inference about a corpus that no longer exists, indistinguishable on
    # screen from inference about the one that does.
    try:
        if graph.available():
            graph.clear_links()
    except Exception as exc:
        log.error("the inferred edges could not be removed from the graph: %s",
                  exc)
    if count:
        log.info("discarded %d entity link judgement(s): the corpus changed",
                 count)
    return count


# -- the run -----------------------------------------------------------------------

def run(mode: str = "grouped", wanted: list[str] | None = None,
        keep_fragments: bool = False,
        run_id: str = "") -> Iterator[tuple[str, Any]]:
    """Compare entities and record what connects them.

    Yields (kind, data) for the job runner, the same shape report generation
    uses, so this gets the same stoppable, refresh-surviving treatment for free.
    """
    model = llm_settings.effective_text_model()
    if not model:
        yield "error", ("No text model is set, so nothing can be compared. "
                        "Choose one on the settings page.")
        return

    by_type = corpus_entities()
    usable, skipped = _usable(by_type, keep_fragments)
    if not any(usable.values()):
        yield "error", ("There are no entities to compare yet. Upload "
                        "documents and let them finish processing first.")
        return
    for etype, count in skipped.items():
        yield "token", (
            f"{count} of {len(by_type[etype])} {etype} entities were left out: "
            f"they are names of things rather than assertions about them, so "
            f"there is nothing in them to corroborate or contradict.\n\n")

    client = Ollama()
    yield "status", f"Checking {model} is loaded"
    try:
        client.require_model(model, llm_settings.text_model_label())
    except Exception as exc:
        yield "error", str(exc).splitlines()[0]
        return

    options = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                              "LINK_NUM_PREDICT", 1200, seed=random_seed())
    names = wanted or sorted(PAIRINGS)
    done = judged_pairs()
    linked = judged = 0

    def show(verdict: dict) -> str:
        a, b = verdict["a"], verdict["b"]
        if verdict.get("subject") == "B":
            a, b = b, a
        return (f"{a['name'][:70]}\n  {verdict['relation']} "
                f"({verdict['confidence']:.2f})\n{b['name'][:70]}\n"
                f"  {verdict['basis']}\n\n")

    for name in names:
        pairing = PAIRINGS.get(name)
        if pairing is None:
            yield "error", f"There is no pairing called {name}."
            continue
        pairs = [p for p in pairs_for(pairing, usable, mode)
                 if tuple(sorted((p[0]["id"], p[1]["id"]))) not in done]
        if not pairs:
            continue
        yield "token", f"\n\n===== {name} — {len(pairs)} pair(s) =====\n\n"

        if mode == "grouped":
            same = pairing.a == pairing.b
            group_a = usable.get(pairing.a, [])
            group_b = group_a if same else usable.get(pairing.b, [])
            plan = windows(group_a, group_b, same)
            for index, (left, right) in enumerate(plan, 1):
                covered = [p for p in window_pairs(left, right, same)
                           if tuple(sorted((p[0]["id"], p[1]["id"]))) not in done]
                if not covered:
                    continue
                yield "status", (f"{name}: window {index} of {len(plan)} "
                                 f"({len(covered)} pair(s))")
                try:
                    verdicts, complete = judge_window(
                        client, model, pairing, left, right, options)
                except Exception as exc:
                    log.error("%s: window %d failed: %s", name, index, exc)
                    yield "error", f"{name}: window {index} failed — {exc}"
                    continue
                if not complete:
                    # The model never said it had finished, so this window's
                    # silence about a pair means nothing. Recording the pairs as
                    # judged would bury hundreds at a time.
                    log.warning("%s: window %d gave no completion marker; its "
                                "%d pair(s) are left for a later run",
                                name, index, len(covered))
                    yield "error", (f"{name}: window {index} did not finish; "
                                    f"its {len(covered)} pair(s) were not judged")
                    continue
                named = {tuple(sorted((v["a"]["id"], v["b"]["id"]))): v
                         for v in verdicts}
                for a, b in covered:
                    key = tuple(sorted((a["id"], b["id"])))
                    verdict = named.get(key) or {"a": a, "b": b,
                                                 "relation": NONE}
                    judged += 1
                    if record(verdict, pairing, model, run_id):
                        linked += 1
                        yield "token", show(verdict)
            continue

        for start in range(0, len(pairs), BATCH_PAIRS):
            batch = pairs[start:start + BATCH_PAIRS]
            yield "status", (f"{name}: pair {start + 1}-"
                             f"{start + len(batch)} of {len(pairs)}")
            try:
                verdicts = judge(client, model, pairing, batch, options)
            except Exception as exc:
                # One batch failing is not the run failing. The pairs in it are
                # left unjudged, so a later run picks them up exactly because
                # nothing was written for them.
                log.error("%s: batch at %d failed: %s", name, start, exc)
                yield "error", f"{name}: a batch failed — {exc}"
                continue
            for verdict in verdicts:
                judged += 1
                if record(verdict, pairing, model, run_id):
                    linked += 1
                    yield "token", show(verdict)

    yield "status", "Drawing the connections into the graph"
    try:
        drawn = draw()
    except Exception as exc:
        log.error("the graph could not be redrawn: %s", exc)
        yield "error", f"the connections were recorded but not drawn: {exc}"
        drawn = 0
    yield "token", (f"\n\n{judged} pair(s) judged, {linked} connection(s) "
                    f"recorded, {drawn} drawn into the graph.\n")
    yield "done", {"judged": judged, "links": linked, "drawn": drawn}
