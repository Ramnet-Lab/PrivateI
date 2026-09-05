#!/usr/bin/env python3
"""Score a processed case against a ground-truth key.

The key format is the contract, not this case: any investigation with a key in
the same schema scores the same way. What is measured is deliberately the
mechanics of a defensible investigation rather than prose quality -

    entity_resolution   precision must be 1.0 (a phantom person is a hard fail)
    negation_accuracy   polarity traps kept their sign
    date_accuracy       no precision upgrades (an approximation asserted as a day)
    sourcing_discipline documentary facts cite their custodian, not a restatement
    fact_recall         gold facts present with a correct citation
    merge_integrity     entity merges terminate - no cycle, no dangling target
    assertion_dedup     one sentence quoted at four lengths is one fact
    date_coverage       an assertion whose own text names a date carries one
    conflict_recall     conflicts detected (report mode only)
    conflict_precision  each reported conflict is two accounts that oppose
    corpus_completeness every ingested document reached the report
    citation_integrity  every finding cites a document, not the allegation
    quote_routing       a finding rests on evidence bearing on its allegation
    disposition_enum    every disposition is one of the labels the key names
    disposition_accuracy per allegation, and reached by weighing the
                        elements rather than defaulted to when the weighing
                        failed, plus summary/findings agreement

    python3 eval/score_case.py eval/keys/cdi_2026-04.json
    python3 eval/score_case.py <key> --report reports/xyz.md   # also grade a report

Exit code is the number of failed thresholds.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rows(sql: str, params=()) -> list[dict]:
    conn = sqlite3.connect(f"file:{ROOT/'data'/'state.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def threshold(key: dict, category: str, default: int) -> int:
    """Read the pass bar out of the key. Hardcoding it here would mean the
    harness quietly grades every case against whatever the last case needed.
    """
    raw = str(key.get("scoring", {}).get(category, {}).get("pass_threshold", ""))
    m = re.search(r"(\d+)\s*/\s*\d+", raw)
    return int(m.group(1)) if m else default


def ratio_threshold(key: dict, category: str, default: float) -> float:
    """The pass bar for a check whose result is a share rather than a count.

    A key author has no reason to know which check parses their number, so
    read every form a bar is plausibly written in - "13/29", "45%", "0.45" -
    rather than making one spelling the only one that is heard.
    """
    raw = str(key.get("scoring", {}).get(category, {}).get("pass_threshold", ""))
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", raw)
    if m and float(m.group(2)) > 0:
        return float(m.group(1)) / float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", raw)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"\d*\.\d+", raw)
    if m:
        return float(m.group(0))
    return default


def tuning(key: dict, name: str, default: float) -> float:
    """Read one grading constant out of the key, defaulting to the value here.

    threshold() reads the pass bar for a whole category; this reads the smaller
    numbers an individual check turns on - how much of a gold fact a
    paraphrase has to keep before it counts as tracing to that fact, how much
    of one account has to reappear in another before the two are one statement
    told twice. Leaving those in the source has the same defect as leaving a
    pass bar there: the number stops describing the case being graded and
    starts describing the corpus whose prose style it was tuned against.
    """
    raw = key.get("scoring", {}).get("tuning", {}).get(name)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


# The report is a structured document, so read it as one. Every check that
# talks about "a finding under allegation 2" needs the same three things:
# which allegation a passage sits under, which subsection of that allegation
# it is in, and what it cites. Parsing that once stops five checks from each
# inventing a slightly different idea of where an allegation begins.
#
# The document is written by a markdown-trained model, so each of those
# anchors arrives wearing decoration nobody pictured when the anchor was
# written: bold on both sides of a colon, a subsection heading hidden behind
# a trailing colon, a heading written as a bullet or a numbered line, a
# summary row that is a pipe table whose cells carry bold of their own.
# Patching one regex per variation as it turns up has been the largest single
# defect class in this harness, so decoration is read off once, in md_lines
# below, and every anchor in this file is matched against a line's undecorated
# text rather than against one spelling of it.
CITATION = re.compile(r"\[([^\]\[]{1,160})\]")
QUOTED = re.compile(r"[\"“]([^\"“”]{12,}?)[\"”]")
NEG = re.compile(r"\b(never|not|n't|no|zero|none|failed|denied|refused|"
                 r"without|absent|missing|unable)\b", re.I)
# Function words carry no position. Left in, "the" and "of" alone would make
# any two sentences look like restatements of each other.
STOPWORDS = frozenset(
    "the a an and or but of to in on at for from with that this it its as by "
    "was were is are be been being he she they them his her their our my me i "
    "you we who whom which there here then than so".split())


def flat(s: str) -> str:
    """One spelling for NOT_SUBSTANTIATED, Not substantiated and not-substantiated."""
    return re.sub(r"[^a-z]+", " ", str(s).lower()).strip()


# Asterisk and backtick runs are always decoration in a report. Underscores
# are not - an ingested filename is full of them, and stripping those would
# leave a citation matching no document - so an underscore counts as emphasis
# only where it sits on the outer edge of a word.
_MARKS = re.compile(r"\*{1,3}|`{1,3}|~~")
_EDGE_UNDERSCORE = re.compile(r"(?<![0-9A-Za-z_])_{1,3}(?=[^\s_])"
                              r"|(?<=[^\s_])_{1,3}(?![0-9A-Za-z_])")
# What a line may wear in front of its content: a blockquote arrow, a
# heading's hashes, a bullet in any of markdown's spellings, or a list number.
_LEADER = re.compile(r"^(?:\s*>)*\s*(?:#{1,6}\s+|[-*+•]\s+|\(?\d+[.)]\s+)?")
_ITEM_LEADER = re.compile(r"^[ \t]*(?:[-*+•]|\(?\d+[.)])[ \t]+")
_HEADING_LEADER = re.compile(r"^[ \t]*(#{1,6})[ \t]+")
# A line that is nothing but decoration wrapped around its content is a label
# whatever it says - "**Findings**", "**Gaps:**", "*note*" - because a
# sentence does not usually put emphasis around the whole of itself.
_WRAPPED = re.compile(r"^(?:\*{1,3}|_{1,3})(.+?)(?:\*{1,3}|_{1,3}):?$")
_DELIM_CELL = re.compile(r"^:?-{2,}:?$")


def undecorate(s: str) -> str:
    """One line's text with its markdown decoration read off.

    Bold, italics, code spans, a heading's hashes, a bullet or list number and
    a blockquote arrow are packaging rather than content, so they come off
    before anything in this file compares a line to an anchor. Whitespace is
    collapsed at the same time: a decoration removed from the middle of a line
    leaves a gap that would otherwise defeat an exact comparison.
    """
    s = _MARKS.sub("", s)
    s = _EDGE_UNDERSCORE.sub("", s)
    s = _LEADER.sub("", s, count=1)
    return " ".join(s.split())


def _cells(raw: str) -> list[str]:
    """A table row's cells, each undecorated. The outer pipes are packaging."""
    return [undecorate(c) for c in raw.strip().strip("|").split("|")]


def md_lines(text: str) -> list[dict]:
    """Every line of a document, read as markdown once.

    Each entry carries where the line sits in the source, what it says with
    its decoration removed, its cells when it is a table row, and the facts an
    anchor needs about its packaging: heading depth, whether it is a list
    item, how far it is indented, and whether the whole line is wrapped in
    emphasis. Everything in this file that looks for structure goes through
    here, so a spelling of a decoration is handled in one place rather than in
    nine regexes that each learned about it separately.
    """
    out: list[dict] = []
    pos = 0
    for raw in text.split("\n"):
        start, pos = pos, pos + len(raw) + 1
        stripped = raw.strip()
        row = stripped.startswith("|") and stripped.count("|") >= 2
        # A row too wide for one line arrives as two, the second carrying the
        # rest of the cells and no leading pipe. Joined back on, the verdict is
        # in a column again; left split it is in no column at all.
        if (not row and out and out[-1]["cells"] is not None and stripped
                and "|" in stripped
                and not out[-1]["raw"].rstrip().endswith("|")):
            prev = out[-1]
            prev["raw"] = prev["raw"].rstrip() + " " + stripped
            prev["cells"] = _cells(prev["raw"])
            prev["text"] = " ".join(c for c in prev["cells"] if c)
            prev["end"] = start + len(raw)
            continue
        head = _HEADING_LEADER.match(raw)
        cells = _cells(raw) if row else None
        out.append({
            "raw": raw,
            "start": start,
            "end": start + len(raw),
            "text": (" ".join(c for c in cells if c) if cells is not None
                     else undecorate(raw)),
            "cells": cells,
            "heading": len(head.group(1)) if head else 0,
            # A list number can itself be decorated - "**1.** the subject
            # stated" is a numbered entry - so the leader is looked for on the
            # line with its emphasis removed as well as on the line as written,
            # where an undecorated bullet still lives.
            "item": bool(_ITEM_LEADER.match(raw)
                         or _ITEM_LEADER.match(_MARKS.sub("", raw))),
            "indent": len(raw) - len(raw.lstrip()),
            "wrapped": bool(_WRAPPED.match(stripped)),
        })
    return out


def line_label(line: dict) -> tuple[str, str] | None:
    """(label, value) for a line that announces one, else None.

    "**Disposition:** **Not substantiated**", "Disposition: Not substantiated",
    "#### Disposition", "- **Findings:**" and "| Disposition | Not
    substantiated |" are one anchor wearing five coats, and the value is
    whatever the line says after the label - decoration on the far side of the
    colon included, which is where several of these checks used to lose it.
    """
    cells = line["cells"]
    if cells is not None:
        named = [c.strip() for c in cells]
        if not named or not named[0] or named[0][0].isdigit():
            return None
        return (named[0].rstrip(":").strip(),
                " ".join(c for c in named[1:] if c).strip())
    text = line["text"]
    if not text:
        return None
    m = re.match(r"^([^:]{1,60}?)\s*:\s*(.*)$", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if line["heading"] or line["wrapped"]:
        return text, ""
    return None


def label_value(line: dict, name: str) -> str | None:
    """What this line puts against `name`, or None when it is not that line.

    The match is a prefix one because the anchor and the template's wording
    for it are not the same string: "Conflicts" has to reach "Conflicts in the
    evidence". A line with no colon at all is still read - a model writing
    "**Disposition** Not substantiated" has answered the question - but only
    where it is not a list item, since a finding beginning with the same word
    is a finding rather than a label.
    """
    found = line_label(line)
    if found is not None and found[0].lower().startswith(name.lower()):
        return found[1]
    if line["item"]:
        return None
    m = re.match(rf"(?i)^{re.escape(name)}\b[^0-9A-Za-z]*(.*)$", line["text"])
    return m.group(1).strip() if m else None


def labelled(lines: list[dict], name: str) -> list[tuple[int, str]]:
    """Every line announcing `name` as (index, value), exact matches first.

    A prefix match has to reach "Conflicts in the evidence" from "Conflicts",
    which also means it reaches a note reading "Disposition corrected at
    generation: ..." from "Disposition". A line naming the anchor and nothing
    else is therefore preferred over one that carries on into a sentence, so
    that the line every reader parses is the line this file parses too.
    """
    exact: list[tuple[int, str]] = []
    prefix: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        value = label_value(line, name)
        if value is None:
            continue
        head = (line_label(line) or (line["text"], ""))[0]
        (exact if flat(head) == flat(name) else prefix).append((i, value))
    return exact + prefix


def _flat(text: str) -> str:
    return " ".join(str(text or "").lower().split()).strip(":*# ")


def _opens_label(line: dict) -> bool:
    """Whether a line starts something new rather than continuing a list.

    A subsection ends where the report starts talking about something else:
    a line announcing a label and saying nothing else - "**Conflicts in the
    evidence**", "## Gaps", "Gaps:" - or a line putting a short one-word label
    against a value, which is how a disposition line ends a findings list that
    ran before it. A table row never ends one: a findings list written as a
    table would otherwise be cut off by its own header.
    """
    if line["cells"] is not None:
        return False
    found = line_label(line)
    if found is None:
        return False
    label, value = found
    if line["item"] and value:
        return False
    if not label or len(label.split()) > 6:
        return False
    # A conflict entry is built out of labelled lines - TYPE, EVENT, A, B,
    # INCOMPATIBLE - and every one of them is a short label against a value.
    # Ending the section at the first of them truncated every conflict body to
    # its opening line, so the checks read a report full of conflicts as a
    # report with none. Only a label that names a section the report actually
    # has ends one.
    if value and _flat(label) not in SECTION_LABELS:
        return False
    return not value or len(label.split()) == 1


# The sections a report announces. Anything else labelled is content.
SECTION_LABELS = {
    "findings", "findings of fact", "conflicts", "conflicts in the evidence",
    "corroboration", "gaps", "disposition", "dispositions", "elements",
    "elements and weighing", "basis", "summary", "timeline", "persons",
    "persons named",
}


_ALLEGATION = re.compile(
    r"(?i)^allegation\s+(\d+)\s*(?:([:.\-–—)])\s*)?(.*)$")


def allegation_head(line: dict) -> tuple[int, str] | None:
    """(number, title) when this line opens an allegation's section.

    Reading every decorated spelling of the heading is not the same as reading
    every line that mentions an allegation. "1. Allegation 2 is supported by
    the same email" is a finding, and treating it as a heading would file
    everything after it under a section that never started. A heading is
    therefore a line packaged as one - a markdown heading, or a line that is
    nothing but decoration around the anchor - or a line that puts a title
    after the number with a separator between them. A summary row mentioning
    an allegation is data about the section rather than the section itself.
    """
    if line["cells"] is not None and len([c for c in line["cells"] if c]) > 1:
        return None
    m = _ALLEGATION.match(line["text"])
    if not m:
        return None
    sep, title = m.group(2), m.group(3).strip()
    if not (line["heading"] or line["wrapped"] or sep or not title):
        return None
    return int(m.group(1)), title


def allegation_blocks(text: str) -> dict[int, dict]:
    """{number: {"title": the report's restatement, "body": everything under it}}."""
    lines = md_lines(text)
    marks = [(i, head) for i, line in enumerate(lines)
             if (head := allegation_head(line)) is not None]
    out: dict[int, dict] = {}
    for k, (i, (number, title)) in enumerate(marks):
        stop = lines[marks[k + 1][0]]["start"] if k + 1 < len(marks) else len(text)
        out[number] = {"title": title, "body": text[lines[i]["end"]:stop]}
    return out


def subsection(body: str, name: str) -> str:
    """One subsection of an allegation block - Findings, Conflicts, Gaps.

    The subsection runs from its heading to the next thing the report
    announces, in whatever way it announces either. Requiring the heading to
    be bold read "**Findings**" and nothing else: not "## Findings", not
    "Findings:", and not "**Findings:**", each of which is the same heading
    with a different coat on and each of which returned an empty section that
    every check downstream then scored as a report containing no findings.
    """
    lines = md_lines(body)
    found = labelled(lines, name)
    if not found:
        return ""
    i, value = found[0]
    stop = len(body)
    for line in lines[i + 1:]:
        if line["heading"] or _opens_label(line):
            stop = line["start"]
            break
    return (value + "\n" + body[lines[i]["end"]:stop]).strip()


def numbered_items(section: str) -> list[str]:
    """The numbered entries of a subsection, one string each.

    Markdown has several spellings of a list and a model uses all of them:
    "1.", "1)", a dash, a bullet, or a table with the number in its first
    column. An indented list under an entry belongs to that entry rather than
    standing as a new one, and a line that starts no entry continues the entry
    it follows, which is how an entry long enough to wrap stays one entry.
    """
    items: list[list[str]] = []
    for line in md_lines(section):
        cells = line["cells"]
        if cells is not None:
            body = [c for c in cells if c]
            if not body or all(_DELIM_CELL.match(c) for c in body):
                continue
            if re.fullmatch(r"\d+", body[0]) and len(body) > 1:
                items.append([" ".join(body[1:])])
            elif items:
                items[-1].append(" ".join(body))
            continue
        if line["item"] and line["indent"] < 2:
            items.append([line["text"]])
        elif items and line["text"]:
            items[-1].append(line["text"])
    return [text for text in (" ".join(parts).strip() for parts in items) if text]


# A double underscore before a hex run is unambiguous, so six characters are
# enough there. A single separator is not - "facade" and "decade" are ordinary
# words that happen to be hexadecimal - so only a run long enough to be a real
# content hash counts as one.
HASH_SUFFIX = re.compile(r"(?:__[0-9a-f]{6,}|[-_.][0-9a-f]{8,})$")


def doc_stems(docs: list[dict], key: dict) -> dict[str, str]:
    """The distinctive part of every ingested document's name.

    Ingest appends a content hash to the filename, so the stem before it is
    what a citation and the key's file_match have in common. Matching on
    anything shorter - the leading id alone - would let one document's
    citation stand in for another whose name merely starts the same way.

    The hash is stripped only where one is actually present, and the
    separator is read rather than assumed: a naming scheme that joins the
    hash with a single underscore or a dash is still a naming scheme, and
    assuming a double underscore returned stems that no citation contains,
    which fails corpus completeness and citation integrity wholesale on a
    correct report.
    """
    if not docs:
        docs = [{"doc_id": d.get("doc_id"), "filename": d.get("file_match", "")}
                for d in key.get("documents", [])]
    out: dict[str, str] = {}
    for d in docs:
        name = str(d.get("filename") or d.get("doc_id") or "")
        stem = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", name)
        trimmed = HASH_SUFFIX.sub("", stem)
        stem = trimmed or stem
        if stem:
            out[str(d.get("doc_id") or name)] = norm(stem)
    return out


VALUE = re.compile(r"\b\d+\b"
                   r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
                   r"|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b")


def _content(s: str) -> set[str]:
    return {w for w in norm(s).split() if len(w) > 2 and w not in STOPWORDS}


def _values(s: str) -> set[str]:
    """Numbers, months and weekdays: the particulars two accounts can differ on
    without either being a paraphrase of the other."""
    out = set()
    for m in VALUE.finditer(norm(s)):
        token = m.group(0)
        out.add(token if token.isdigit() else token[:3])
    return out


def _denied(s: str, shared: set[str]) -> set[str]:
    """Which of the other side's words this side negates.

    A negation standing next to a word the other side asserts is a
    contradiction. A negation about something the other side never mentions -
    "no pressure" in an account that says nothing about pressure - is a
    separate remark that happens to sit in the same sentence.
    """
    words = norm(s).split()
    out: set[str] = set()
    for i, w in enumerate(words):
        if NEG.fullmatch(w):
            out |= {x for x in words[i + 1:i + 5] if x in shared}
    return out


# How much of the shorter account has to reappear in the longer one before the
# two are treated as the same statement told twice. Set high on purpose: a
# loose paraphrase slips through and is scored as a real conflict, which is a
# miss rather than a false accusation. This is the default; a key whose
# reports paraphrase more or less tightly overrides it under scoring.tuning.
OPPOSE_CONTAINMENT = 0.6


def opposes(a: str, b: str, containment: float = OPPOSE_CONTAINMENT) -> bool:
    """Whether two accounts of the same thing cannot both be true.

    Three ways they cannot: they put different numbers, months or weekdays in
    play; one denies something the other asserts; or their content does not
    substantially overlap, which is the ordinary case of two witnesses
    describing incompatible scenes. What is left over - one account repeating
    most of the other's words, with the same particulars and the same polarity
    - is corroboration written up as a disagreement.
    """
    va, vb = _values(a), _values(b)
    if (va or vb) and va != vb:
        return True
    ca, cb = _content(a), _content(b)
    if not ca or not cb:
        return True
    shared = ca & cb
    if _denied(a, shared) != _denied(b, shared):
        return True
    return len(shared) / min(len(ca), len(cb)) < containment


def conflict_positions(item: str) -> list[str]:
    """The accounts one conflict entry sets against each other.

    Each account ends at its citation, because that is where the report stops
    speaking for one source and starts speaking for the next. Where the
    account quotes its source verbatim the quote is the position; the
    surrounding "X stated that" is the report's framing, not the claim.
    """
    parts, last = [], 0
    for m in CITATION.finditer(item):
        chunk = item[last:m.start()]
        last = m.end()
        quotes = QUOTED.findall(chunk)
        text = " ".join(quotes) if quotes else chunk
        text = text.strip(" .,;:-")
        if len(text.split()) >= 3:
            parts.append(text)
    if not parts:
        parts = [q for q in QUOTED.findall(item) if len(q.split()) >= 3]
    return parts


def word_present(word: str, blob: str) -> bool:
    """Is this word in the text, allowing for its ordinary inflections?

    A key writes the event as "APC referral to Hahn" where the document says
    "referred it to his Flight Chief". Substring matching calls that a miss and
    reports a timeline gap the pipeline does not have. Matching on a stem is
    not a loosening: the stem has to be at least five characters and has to
    start a word, so "refer" reaches "referred" and "referral" while staying
    clear of unrelated words.
    """
    # Anchored at a word boundary in both branches. Plain substring matching
    # finds "move" inside "removed" and "was" inside "wasp", which credits the
    # pipeline for events it never recorded.
    # A short word has to match whole. Prefix matching is what lets "move"
    # reach "moved", but the same latitude lets "was" reach "wasp", and a
    # three-letter word carries too little signal to spend that on.
    if len(word) < 4:
        return re.search(rf"\b{re.escape(word)}\b", blob) is not None
    if re.search(rf"\b{re.escape(word)}", blob):
        return True
    stem = re.sub(r"(?:als?|ments?|ings?|ed|es|s)$", "", word)
    if len(stem) < 5:
        return False
    return re.search(rf"\b{re.escape(stem)}\w*", blob) is not None


def score_entities(key: dict, facts: list[dict]) -> dict:
    """Precision is the one that must be perfect: inventing a person is worse
    than missing one, because a phantom carries other people's actions."""
    found = {f["subject_name"] for f in facts if f["subject_type"] == "PERSON"} | \
            {f["object_name"] for f in facts if f["object_type"] == "PERSON"}
    persons = key["entities"]["persons"]

    matched, unmatched = set(), set()
    for name in found:
        n = norm(name)
        hit = None
        for p in persons:
            forms = [p["canonical"]] + list(p.get("aliases", []))
            if any(norm(f) and (norm(f) in n or n in norm(f)) for f in forms):
                hit = p["id"]
                break
        if hit:
            matched.add(hit)
        else:
            unmatched.add(name)

    # A name appearing in no roster entry is only a phantom if it is also not a
    # peripheral mention the key says is optional.
    peripheral = " ".join(key["entities"].get("peripheral_mentions_not_required", [])).lower()
    phantoms = sorted(n for n in unmatched if norm(n).split()[-1:] and
                      norm(n).split()[-1] not in peripheral)

    precision = 1.0 - (len(phantoms) / max(1, len(found)))
    return {"precision": round(precision, 3), "recall": f"{len(matched)}/{len(persons)}",
            "phantoms": phantoms,
            "pass": not phantoms and len(matched) >= len(persons) - 1}


def score_negation(key: dict, facts: list[dict]) -> dict:
    """Each negation trap must appear with its negation intact.

    Matching uses the gold fact the trap names where there is one, because a
    trap is written for a human ("Pike explicitly did NOT see Morgan's screen")
    while the document says "I never saw his screen" - overlapping on the
    trap's own wording missed a fact that was extracted perfectly.
    """
    traps = key.get("trap_inventory", {}).get("negation", [])
    gold = {g["id"]: g["text"] for g in key.get("gold_facts", [])}

    # Words that describe the trap rather than appear in the evidence.
    META = {"explicitly", "implies", "implied", "after", "which", "means"}

    results = []
    for trap in traps:
        fid = re.search(r"\bF\d{2}\b", trap)
        source = gold.get(fid.group(0), trap) if fid else trap
        words = [w for w in norm(source).split()
                 if len(w) > 3 and w not in META][:8]
        if not words:
            results.append({"trap": trap[:52], "status": "unscorable"})
            continue

        blob_of = lambda f: norm(f["predicate"] + " " + f["object_name"] + " " +
                                 f["subject_name"] + " " + f["quote"])
        # Overlap alone is not identity. Requiring two shared words missed
        # "his screen" against "Morgan's screen"; accepting one matched any
        # negated fact that happened to share a common word like "checks".
        # Weight each shared word by how rare it is across the whole case, so
        # "screen" identifies a fact and "checks" does not.
        blobs = [blob_of(f) for f in facts]
        def weight(word: str) -> float:
            seen = sum(1 for b in blobs if word in b)
            return 0.0 if seen == 0 else 1.0 / seen
        scored = []
        for f, blob in zip(facts, blobs):
            if not NEG.search(blob):
                continue
            score = sum(weight(w) for w in words if w in blob)
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda x: -x[0])
        # Accept the best negated candidate when it shares something genuinely
        # distinctive - a word appearing in at most ten facts - or a broad
        # agreement of three common words. A flat score threshold rejected a
        # correct match whose only shared words were case-wide names.
        def distinctive(f) -> bool:
            blob = blob_of(f)
            shared = [w for w in words if w in blob]
            return (any(sum(1 for b in blobs if w in b) <= 10 for w in shared)
                    or len(shared) >= 3)
        related = [f for _sc, f in scored if distinctive(f)]
        if related:
            results.append({"trap": trap[:52], "status": "kept",
                            "as": related[0]["predicate"][:32]})
            continue
        # Polarity is only lost when the SAME actor is asserted to have done the
        # thing. "Morgan never replied" alongside "Voss replied" is two correct
        # facts about two people; reading the second as the first losing its
        # negation accused the pipeline of the exact error it had avoided.
        actors = {w for w in words
                  if any(w in norm(f["subject_name"]) for f in facts)}
        positive = [f for f in facts
                    if sum(w in blob_of(f) for w in words) >= 2
                    and (not actors or any(a in norm(f["subject_name"]) for a in actors))]
        results.append({"trap": trap[:52],
                        "status": "POLARITY LOST" if positive else "not found"})

    lost = [r for r in results if r["status"] == "POLARITY LOST"]
    missing = [r for r in results if r["status"] == "not found"]
    return {"traps": results, "kept": len(results) - len(lost) - len(missing),
            "of": len(results), "polarity_lost": len(lost),
            "not_extracted": [r["trap"][:40] for r in missing],
            "pass": not lost and not missing}


def score_dates(key: dict, facts: list[dict]) -> dict:
    """The failure that matters is a precision upgrade: an approximation or a
    month rendered as a specific day and presented as stated fact."""
    upgrades = []
    for f in facts:
        date, basis = f.get("event_date"), (f.get("event_date_basis") or "")
        if not date or len(date) < 10:
            continue
        text = norm(f["quote"])
        # The hedge has to modify a DATE, not anything else in the sentence.
        # "about a foot from my face" and "about fifteen feet away" are
        # distances; an earlier version of this check read them as approximate
        # dates and reported three failures that were entirely its own.
        approx = re.search(
            r"\b(around|about|approximately|mid|early|late|sometime|following)\b"
            r"(?:\W+\w+){0,2}?\W+"
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|monday|tuesday|"
            r"wednesday|thursday|friday|saturday|sunday|week|month|\d{4})",
            text)
        # "since March" (no day) is month precision; "since 18 March" is exact.
        vague_since = re.search(
            r"\bsince\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
            r"(?!\s+\d)", text)
        if (approx or vague_since) and basis in ("", "stated"):
            upgrades.append(f"{f['predicate']} @ {date} <- {f['quote'][:60]!r}")
    # Zero upgrades is only half the threshold; the other half is that the
    # dated events actually reached the timeline WITH their dates. Keyword
    # overlap alone cannot tell the first of four like-worded events from the
    # second, third and fourth - one extracted event would satisfy all four
    # gold entries and report recall that was never earned. A timeline is a claim
    # about when, so the date has to match to the precision the key states.
    gold_tl = key.get("timeline_gold", [])
    found, missed = 0, []
    for event in gold_tl:
        words = [w for w in norm(event["event"]).split() if len(w) > 3][:5]
        if not words:
            continue
        need = max(2, len(words) // 2)
        gold_date = str(event.get("date") or "")
        width = 7 if event.get("precision") == "month" else 10
        hit = False
        for f in facts:
            blob = norm(f["quote"] + " " + f["predicate"] + " " +
                        str(f["object_name"]) + " " + str(f["subject_name"]))
            if sum(word_present(w, blob) for w in words) < need:
                continue
            if not gold_date:
                hit = True
                break
            got = str(f.get("event_date") or "")
            if got[:width] == gold_date[:width]:
                hit = True
                break
        found += 1 if hit else 0
        if not hit:
            missed.append(f"{gold_date} {event['event'][:44]}")

    need_tl = threshold(key, "date_accuracy", default=18)
    return {"precision_upgrades": upgrades[:6], "upgrade_count": len(upgrades),
            "timeline_events": f"{found}/{len(gold_tl)}", "needed": need_tl,
            "missed": missed[:8],
            "pass": not upgrades and found >= need_tl}


def score_sourcing(key: dict, facts: list[dict]) -> dict:
    """A documentary figure must be cited to the document of record.

    Matching is done on the GOLD FACT the trap names (F15/F16...), never on the
    trap's own prose. An earlier version took content words from the trap
    sentence itself - 'primary', 'source', 'restatement' - and so matched any
    quote containing the word "source". That produced a pass while testing the
    wrong document, and then a failure while testing nothing at all. The trap
    text describes the check; the gold facts are the check.
    """
    docs = {d["doc_id"]: d["file_match"] for d in key["documents"]}
    gold = {g["id"]: g for g in key["gold_facts"]}
    results = []

    for trap in key.get("trap_inventory", {}).get("sourcing", []):
        # Keys identify documents differently ("doc 09", "D09"); read
        # whichever form this key uses rather than assuming one. The leading
        # "doc" is dropped only when what follows it starts the id, because a
        # key whose ids are themselves spelled "DOC_B" otherwise has the DOC
        # eaten off the front and every citation then fails to match a
        # document that was cited correctly.
        m = re.search(r"primary source is (?:doc(?:ument)?[\s#]*(?=[A-Za-z0-9]))?"
                      r"([A-Za-z0-9_]+)", trap, re.I)
        if not m:
            continue
        primary = m.group(1)
        if primary.isdigit():
            primary = primary.zfill(2)
        fact_ids = re.findall(r"\bF\d{2}\b", trap)
        if not fact_ids:
            continue

        cited_docs: set[str] = set()
        matched_ids: list[str] = []
        for fid in fact_ids:
            g = gold.get(fid)
            if not g:
                continue
            words = [w for w in norm(g["text"]).split() if len(w) > 4][:6]
            if not words:
                continue
            need = max(2, len(words) // 3)
            hits = [f for f in facts
                    if sum(w in norm(f["quote"] + " " + f["object_name"]) for w in words) >= need]
            if hits:
                matched_ids.append(fid)
                cited_docs |= {f["doc_id"] for f in hits}

        ok = bool(cited_docs) and any(c.startswith(primary + "_") for c in cited_docs)
        results.append({"facts": fact_ids, "matched": matched_ids,
                        "primary_doc": docs.get(primary, "")[:26],
                        "cited_primary": ok,
                        "cited": sorted(c[:24] for c in cited_docs)[:3]})

    checked = [r for r in results if r["matched"]]
    return {"checks": results,
            "note": "unmatched traps are a recall gap, not a sourcing error",
            "pass": bool(checked) and all(r["cited_primary"] for r in checked)}


def score_facts(key: dict, facts: list[dict]) -> dict:
    """A gold fact counts as recalled when its distinctive content appears in a
    quote from one of the documents the key says carries it."""
    docs = {d["doc_id"]: d["file_match"] for d in key["documents"]}
    recalled, missed = [], []
    for gold in key["gold_facts"]:
        words = [w for w in norm(gold["text"]).split() if len(w) > 4][:6]
        if not words:
            continue
        want_docs = [docs.get(gold["primary_doc"], "")] + \
                    [docs.get(d, "") for d in gold.get("corroborating_docs", [])]
        hit = False
        for f in facts:
            blob = norm(f["quote"] + " " + f["predicate"] + " " + f["object_name"])
            if sum(w in blob for w in words) >= max(2, len(words) // 3):
                if any(w and w.lower()[:12] in f["doc_id"].lower() for w in want_docs if w):
                    hit = True
                    break
        (recalled if hit else missed).append(gold["id"])
    thresh = key.get("scoring", {}).get("fact_recall", {}).get("pass_threshold", "")
    m = re.search(r"(\d+)\s*/\s*(\d+)", thresh)
    need = int(m.group(1)) if m else max(1, int(len(key["gold_facts"]) * 0.78))
    return {"recalled": len(recalled), "total": len(key["gold_facts"]),
            "needed": need, "missed": missed, "pass": len(recalled) >= need}


# Words that carry no signal in a position description. Short words are kept
# otherwise: "duty", "paid", "June" and "days" each decide something here.
POSITION_STOPWORDS = {
    "the", "and", "but", "for", "with", "that", "this", "from", "into", "over",
    "under", "were", "was", "been", "have", "has", "had", "not", "only",
    "than", "then", "them", "they", "their", "there", "here", "when", "what",
    "which", "who", "whom", "would", "could", "should", "about", "after",
    "before", "both", "each", "more", "most", "some", "such", "very",
}


def score_conflicts(key: dict, text: str) -> dict:
    """Which expected conflicts the report actually surfaced.

    A conflict counts as detected when the report's conflict sections name
    enough of both positions to show it understood the disagreement - not
    merely that the words appear somewhere in the document.
    """
    # Read through the same subsection parser every other report check uses.
    # A bold-only pattern found the section under one spelling of its heading
    # and returned nothing under "## Conflicts in the evidence" or
    # "**Conflicts in the evidence:**", which scores a report that surfaced
    # every conflict as one that surfaced none.
    blocks = allegation_blocks(text)
    sections = [subsection(b["body"], "Conflicts") for b in blocks.values()]
    if not any(s.strip() for s in sections):
        sections = [subsection(text, "Conflicts")]
    blob = norm(" ".join(sections))
    whole = norm(text)
    detected, missed = [], []
    for c in key.get("expected_conflicts", []):
        sides = list(c.get("positions", {}).values())
        if not sides:
            # A conflict the key states without positions - a hearsay chain
            # whose point is that it carries no independent weight - cannot be
            # matched side against side, and scoring it as missed made it
            # impossible to pass however well the report handled it. It is
            # matched instead on what the key says the tool should show, read
            # across the whole report, because that behaviour belongs in the
            # weighing as often as in a conflicts list.
            want = [w for w in norm(c.get("expected_resolution", "")).split()
                    if len(w) > 4][:8]
            got = sum(word_present(w, whole) for w in want)
            (detected if want and got >= max(2, len(want) // 3)
             else missed).append(c["id"])
            continue
        hits = 0
        for side in sides:
            # Words of four letters carry meaning here - duty, days, June,
            # paid - and dropping them left one position of one conflict with
            # no matchable words at all, so that conflict could never be
            # credited however plainly the report stated it.
            words = [w for w in norm(side).split()
                     if len(w) > 3 and w not in POSITION_STOPWORDS][:5]
            if words and sum(word_present(w, blob) for w in words) >= max(
                    1, len(words) // 3):
                hits += 1
        (detected if hits >= 2 else missed).append(c["id"])
    thresh = key.get("scoring", {}).get("conflict_recall", {}).get("pass_threshold", "")
    m = re.search(r"(\d+)\s*/\s*(\d+)", thresh)
    need = int(m.group(1)) if m else 3
    required = re.findall(r"\bC\d\b", thresh) or ["C1"]
    return {"detected": detected, "missed": missed,
            "needed": f"{need} incl {required}",
            "pass": len(detected) >= need and all(r in detected for r in required)}


def score_merges(key: dict, facts: list[dict]) -> dict:
    """Named merge/split traps: an alias that must join, a surname that must not.

    Generalised from the key rather than hardcoded: any two roster people who
    share a surname must stay distinct, and any alias listed for a person must
    resolve to that person.
    """
    persons = key["entities"]["persons"]
    names = {f["subject_name"] for f in facts if f["subject_type"] == "PERSON"} | \
            {f["object_name"] for f in facts if f["object_type"] == "PERSON"}

    # Split: two roster entries sharing a surname must not collapse into one.
    by_surname: dict[str, list] = {}
    for p in persons:
        by_surname.setdefault(norm(p["canonical"]).split()[-1], []).append(p)
    split_ok, split_note = True, []
    for surname, group in by_surname.items():
        if len(group) < 2:
            continue
        seen = {p["id"] for p in group
                if any(norm(p["canonical"]).split()[0] in norm(n) for n in names)}
        if len(seen) < len(group):
            split_ok = False
            split_note.append(f"{surname}: only {sorted(seen)} distinguishable")

    # Merge: an alias must RESOLVE to its person, not merely coexist with them.
    # Checking co-presence passed a case where the alias stood as its own
    # separate entity beside the person it names - the exact split the trap
    # exists to catch.
    canon = {}
    try:
        for r in rows("SELECT entity_id, canonical_name, merged_into FROM entities"):
            canon[norm(r["canonical_name"])] = r["merged_into"]
    except Exception:
        canon = {}
    merge_ok, merge_note = True, []
    for p in persons:
        for alias in p.get("aliases", []):
            a = norm(alias)
            if not a or len(a) < 2 or a.startswith("i ") or "context" in a:
                continue
            if a not in canon:
                continue            # never extracted; that is a recall matter
            if not canon[a]:
                merge_ok = False
                merge_note.append(f"{alias!r} stands alone, not merged into "
                                  f"{p['canonical']}")
    return {"surname_split": "pass" if split_ok else split_note,
            "alias_merge": "pass" if merge_ok else merge_note,
            "pass": split_ok and merge_ok}


def score_merge_graph(key: dict, ents: list[dict]) -> dict:
    """Entity resolution has to terminate.

    A merge edge is the claim that one node's mentions belong to another, so
    the edges have to form a forest: follow merged_into from any node and you
    arrive, in finitely many steps, at a node merged into nothing. That node
    is the person.

    Two nodes merged into each other satisfy every check that asks whether a
    particular alias was merged - the alias IS merged, so score_merges reports
    a pass - while leaving the question the merge was supposed to answer
    unanswerable, because resolution never reaches a canonical row. Downstream
    code either loops or stops at whichever end it entered from, so the same
    person is one entity or two depending on the caller. A merge into a node
    that does not exist fails the same way from the other direction.

    This check names no person and reads no roster; it is a property of the
    graph's shape, so it holds for a case whose people this harness has never
    seen.
    """
    edge = {str(e.get("entity_id")): str(e.get("merged_into") or "")
            for e in ents if e.get("entity_id")}
    cycles: dict[frozenset, str] = {}
    dangling, roots, depth = [], set(), 0
    for start in edge:
        walked, path, node = set(), [], start
        while True:
            if node in walked:
                loop = path[path.index(node):]
                cycles.setdefault(frozenset(loop), " -> ".join(loop + [node]))
                node = ""
                break
            if node not in edge:
                dangling.append(f"{path[-1] if path else start} -> {node} "
                                f"(no such entity)")
                node = ""
                break
            walked.add(node)
            path.append(node)
            if not edge[node]:
                break                      # this node is an unmerged root
            node = edge[node]
        if node:
            roots.add(node)
            depth = max(depth, len(path) - 1)
    merged = [n for n, into in edge.items() if into]
    broken = sorted(cycles.values()) + sorted(dangling)
    # An empty graph is not a clean graph. A run that resolved no entity at all
    # would otherwise satisfy "no cycles" trivially, which is the shape of
    # pass condition these controls exist to keep out of the harness.
    return {"entities": len(edge), "merge_edges": len(merged),
            "roots_reached": len(roots), "longest_chain": depth,
            "cycles": sorted(cycles.values()), "dangling": sorted(dangling),
            "pass": bool(edge) and not broken}


def score_assertion_dedup(key: dict, facts: list[dict]) -> dict:
    """The same statement extracted several times over is not several facts.

    One sentence can yield an assertion for its first clause, another for the
    first two, another for the whole of it: same document, same page, same
    subject, same predicate, quotes that are prefixes of one another. Each is
    the same statement at a different length. Anything that counts how much
    evidence stands behind a finding - corroboration, the weight of one
    account against another, how many sources say a thing - then reads one
    sentence quoted four ways as four sources, and the count is inflated
    exactly where the record is thinnest.

    Grouping is per document and page on purpose. Two documents quoting the
    same sentence are two sources for it, which is corroboration and must not
    be deducted; only a repeat within one passage is a duplicate.
    """
    groups: dict[tuple, list[dict]] = {}
    for f in facts:
        # The event date is part of the identity. One sentence naming several
        # dates for the same conduct is meant to yield one assertion per date -
        # same subject, same predicate, same quote, different day. Those are a
        # series, not a repetition, and grouping without the date reported the
        # correct behaviour as duplication.
        # The object is part of the identity too. Three exhibits taken into
        # custody in one sentence are three facts about three objects, and
        # grouping without the object reported them as one statement repeated -
        # the opposite of what the extractor is documented to do, which is to
        # never fold assertions whose objects differ.
        groups.setdefault((str(f.get("doc_id") or ""), f.get("page_num"),
                           norm(str(f.get("subject_name") or "")),
                           norm(str(f.get("predicate") or "")),
                           norm(str(f.get("object_name") or "")),
                           str(f.get("event_date") or "")), []).append(f)
    duplicates = []
    for gkey, members in groups.items():
        if len(members) < 2:
            continue
        # Shortest first, so the representative of a nested family is the
        # shortest quote in it and every longer one attaches to it rather than
        # each pair being reported twice.
        seen: list[str] = []
        for quote, f in sorted(((norm(str(f.get("quote") or "")), f)
                                for f in members), key=lambda x: len(x[0])):
            host = next((r for r in seen
                         if (r and r in quote) or (not r and not quote)), None)
            if host is None:
                seen.append(quote)
            else:
                duplicates.append(f"{gkey[2][:18]} / {gkey[3][:18]} / "
                                  f"{gkey[4][:18]}: "
                                  f"{str(f.get('quote'))[:56]!r}")
    allowed = int(tuning(key, "max_duplicate_assertions", 0))
    share = len(duplicates) / len(facts) if facts else 0.0
    return {"assertions": len(facts), "duplicates": len(duplicates),
            "duplicate_share": round(share, 3), "allowed": allowed,
            "examples": duplicates[:6],
            "pass": bool(facts) and len(duplicates) <= allowed}


# What makes a sentence datable is that it names a time itself. "may" is a
# month only next to a number - on its own it is the modal verb, and reading
# "I may have signed it" as a dated sentence would swell the denominator with
# sentences that name no time at all and make coverage look worse than it is.
DATABLE = re.compile(
    r"\b(?:jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\bmay\s+(?:\d{1,2}|(?:19|20)\d{2})\b|\b\d{1,2}\s+may\b"
    r"|\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b"
    r"|\b(?:19|20)\d{2}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)


def score_date_coverage(key: dict, facts: list[dict]) -> dict:
    """An assertion whose own quote names a date has to carry that date.

    A timeline is built from event_date, so an assertion extracted out of "on
    14 July the crew stood down" and stored with event_date empty is invisible
    to every question about when. The gap does not show up as an error
    anywhere: the fact is present, its quote is faithful, and nothing is
    wrong with it except that the one field the timeline reads is null. It
    surfaces only as timeline recall, one remove from its cause.

    Only assertions whose text names a time are counted, because an assertion
    about a standing state has no date to carry and should not be charged for
    not having one.
    """
    datable = [f for f in facts if DATABLE.search(str(f.get("quote") or ""))]
    dated = [f for f in datable if str(f.get("event_date") or "").strip()]
    share = len(dated) / len(datable) if datable else 0.0
    # Half is a floor rather than a target, and it is only the default: a key
    # that knows how much of its corpus is dateable states its own bar. The
    # number was chosen without reference to what the run on disk scores, and
    # that run fails it - a default picked to make the present case pass would
    # be the harness grading itself.
    bar = ratio_threshold(key, "date_coverage", default=0.5)
    # Reported, not judged: a date read off the statement's own header is a
    # legitimate inference rather than an invention, and score_dates already
    # tests whether dates are asserted at a precision the text does not carry.
    # Gating on this number here would penalise correct inference.
    inferred = sum(1 for f in facts
                   if str(f.get("event_date") or "").strip()
                   and not DATABLE.search(str(f.get("quote") or "")))
    return {"datable": f"{len(dated)}/{len(datable)}",
            "coverage": round(share, 3), "needed": round(bar, 3),
            "dated_from_context": inferred,
            "undated": [str(f.get("quote"))[:56] for f in datable
                        if not str(f.get("event_date") or "").strip()][:6],
            "pass": bool(datable) and share >= bar}


def _money_forms(value: str) -> list[str]:
    """Every way a document might legitimately write one amount.

    A verbatim transcript says "206 dollars and 45 cents" where the key writes
    $206.45. Both are the same figure faithfully recorded, and a check that
    only knows the digit form reports a fidelity failure against text that is
    perfectly correct.
    """
    raw = value.replace("$", "").replace(",", "").strip()
    forms = {value, raw, value.replace("$", "")}
    if "." in raw:
        whole, cents = raw.split(".", 1)
        cents = (cents + "0")[:2]
        with_comma = f"{int(whole):,}" if whole.isdigit() else whole
        forms |= {
            f"{whole} dollars and {int(cents)} cents",
            f"{with_comma} dollars and {int(cents)} cents",
            f"{whole}.{cents}", f"{with_comma}.{cents}",
        }
    else:
        with_comma = f"{int(raw):,}" if raw.isdigit() else raw
        forms |= {f"{raw} dollars", f"{with_comma} dollars", with_comma}
    return [f.lower() for f in forms if f]


def score_numeric(key: dict, facts: list[dict]) -> dict:
    """Every number the key pins must survive into the facts, in some faithful
    form. Digits, words, and comma grouping all count; a changed digit does not.
    """
    # The key says which gold facts carry the numbers under test ("F10 numeric
    # fields exact"). Sweeping every number in the whole key instead would
    # grade this case against a value set the key never claimed.
    metric = str(key.get("scoring", {}).get("numeric_fidelity", {}).get("metric", ""))
    scoped = set(re.findall(r"\bF\d+\b", metric))
    gold = [g for g in key["gold_facts"]
            if not scoped or g.get("id") in scoped] or key["gold_facts"]
    wanted: list[str] = []
    for g in gold:
        wanted += re.findall(r"\$[\d,]+\.?\d*|\b\d[\d,]{2,}\.?\d*\b", g["text"])
    wanted = sorted(set(wanted))
    blob = " ".join(f["quote"] + " " + f["object_name"] for f in facts).lower()
    found = [w for w in wanted if any(form in blob for form in _money_forms(w))]
    return {"values": f"{len(found)}/{len(wanted)}",
            "missing": [w for w in wanted if w not in found][:6],
            "scope": sorted(scoped) or "all gold facts",
            "pass": len(wanted) == 0 or len(found) == len(wanted)}


def score_exculpatory(key: dict, text: str) -> dict:
    """Evidence against the allegation must reach the analysis of it.

    A tool that only gathers support for the hypothesis it was handed is not
    investigating; the key marks which facts cut the other way.
    """
    gold = {g["id"]: g["text"] for g in key["gold_facts"]}
    blocks = allegation_blocks(text)
    ids, present = [], []
    for trap in key.get("trap_inventory", {}).get("exculpatory_routing", []):
        # The trap names the allegation the fact is exculpatory ON, and that is
        # where it has to appear. Searching the whole report counted a fact
        # that landed under a different allegation as correctly routed, which
        # credits the pipeline for evidence it filed against the wrong question.
        m = re.search(r"\bA(\d+)\b", trap)
        scope = norm(blocks[int(m.group(1))]["body"]) \
            if m and int(m.group(1)) in blocks else norm(text)
        for fid in re.findall(r"\bF\d{2}\b", trap):
            ids.append(fid)
            words = [w for w in norm(gold.get(fid, "")).split() if len(w) > 4][:6]
            if words and sum(w in scope for w in words) >= max(2, len(words) // 3):
                present.append(fid)
    need = threshold(key, "exculpatory_recall", default=max(1, len(ids) - 1))
    return {"exculpatory": f"{len(present)}/{len(ids)}", "found": present,
            "needed": need, "missing": [i for i in ids if i not in present],
            "pass": bool(ids) and len(present) >= need}


def score_credibility(key: dict, text: str) -> dict:
    """Motive/credibility surfaced, and hearsay chains denied probative weight."""
    body = norm(text)
    credibility = any(w in body for w in
                      ("credibility", "motive", "bias", "grievance", "lor",
                       "letter of reprimand", "reliability"))
    # The key asks for a hearsay chain "excluded from probative weight", which
    # is two things: the report has to notice the account is secondhand, and it
    # has to say that costs the account weight. An earlier version tested only
    # the first, as a list of phrasings, and missed a report saying "nothing
    # firsthand ... which limits the weight of his statement" - the very
    # behaviour the metric asks for - because the list expected "not
    # firsthand". Testing both halves is a truer reading of the metric than a
    # longer list of synonyms would be.
    # A report says "hearsay" in more than one way. Naming an account as
    # carrying no independent weight, or as uncorroborated, identifies it as
    # secondhand exactly as "not firsthand" does, and a check that only knows
    # the one vocabulary marks a report that handled the chain correctly as
    # having ignored it. The two halves stay distinct: this half is about
    # recognising the account is not self-standing, the half below about
    # saying that costs it something.
    noticed = any(w in body for w in
                  ("hearsay", "secondhand", "second hand", "firsthand",
                   "first hand", "unnamed", "could not name", "heard about",
                   "told me", "repeating what", "independent weight",
                   "no independent", "uncorroborated", "not corroborated",
                   "learned of it from", "was not present"))
    discounted = any(w in body for w in
                     ("weight", "probative", "cannot rely", "not rely",
                      "limits", "limited", "discount", "corroboration"))
    hearsay = noticed and discounted
    return {"credibility_discussed": credibility,
            "hearsay_noticed": noticed, "hearsay_discounted": discounted,
            "hearsay_flagged": hearsay, "pass": credibility and hearsay}


# A verdict is reached one of two ways: by weighing the elements of the
# allegation, or by defaulting because the weighing could not settle anything.
# The generator says which, per allegation, and these two patterns are how it
# says it - the words it writes when a disposition was never settled, and the
# count of how much of the element table it managed to weigh.
#
# What is deliberately NOT read as a default is the review marker on its own.
# A disposition entailed by a table that was only partly readable carries a
# review marker too, and it was still entailed: one element weighed and failed
# decides an allegation whatever went unread beside it. Refusing credit on the
# marker alone would therefore refuse it only where partial coverage can
# entail anything at all, which is one of the two labels and never the other -
# steering under another name. A default is refused credit; a partial weighing
# that reached a verdict is credited, and neither test reads the verdict.
#
# The default is matched on one word rather than on the sentence the generator
# writes around it: it is the generator's word for that branch, it appears
# nowhere else in an allegation's disposition, and matching the whole sentence
# would tie this file to a phrasing the generator is free to reword.
UNSETTLED = re.compile(r"(?i)\bunsettled\b")
WEIGHED_COUNT = re.compile(r"(?i)\b(\d+)\s+of\s+(\d+)\s+element")


def summary_rows(text: str) -> dict[int, dict]:
    """The dispositions table read by column: {number: {verdict, basis}}.

    The verdict used to be taken as the last bold span in the row, which was
    the right cell only for as long as the verdict was the last decorated
    thing in it. The basis column now carries a bold review marker of its own,
    so the last bold span became that marker and every allegation carrying one
    read as a summary contradicting its own findings - a consistency defect
    reported against a report that did not have one, masking whatever real
    inconsistency the check exists to find. A column is a position, so read it
    as one: the header says which column holds the verdict, and where there is
    no header the third cell is where the template puts it.
    """
    lines = md_lines(text)
    verdict_col: int | None = None
    basis_col: int | None = None
    body: list[dict] = []
    for i, line in enumerate(lines):
        head = [c.strip().lower() for c in (line["cells"] or [])]
        named = [k for k, c in enumerate(head) if c.startswith("disposition")]
        if not named or re.fullmatch(r"\d+", head[0]):
            continue
        verdict_col = named[0]
        basis = [k for k, c in enumerate(head) if c.startswith("basis")]
        basis_col = basis[0] if basis else None
        # A header settles which rows are being read as well as which column
        # holds the verdict: the table ends where the rows stop, and an
        # element table further down the section can carry a number in its
        # first cell too without being a statement about any allegation.
        for row in lines[i + 1:]:
            if row["cells"] is None:
                break
            body.append(row)
        break
    if verdict_col is None:
        body = [line for line in lines if line["cells"] is not None]

    out: dict[int, dict] = {}
    for line in body:
        clean = [c.strip() for c in line["cells"]]
        if not any(clean) or all(not c or _DELIM_CELL.match(c) for c in clean):
            continue
        if not re.fullmatch(r"\d+", clean[0]):
            continue
        column = (verdict_col if verdict_col is not None
                  and verdict_col < len(clean)
                  else min(2, len(clean) - 1))
        rest = (clean[basis_col] if basis_col is not None
                and basis_col < len(clean)
                else " ".join(c for c in clean[column + 1:] if c))
        out[int(clean[0])] = {"verdict": clean[column], "basis": rest}
    return out


def weighing(body: str, basis: str) -> dict:
    """How the disposition in this section was reached, as the report says it.

    A label is not a finding. The generator records, per allegation, how much
    of the element table it managed to weigh and whether the disposition was
    settled by that weighing or defaulted because the weighing settled
    nothing, and it says so in the summary row's basis cell and in the
    paragraph the disposition line sits in. Both are read here. A count of
    zero elements weighed is a default whatever else the section says: nothing
    was weighed, so the verdict came from somewhere other than the weighing.

    Neutral by construction: nothing in here looks at which label was written,
    so a default is identified the same way whichever verdict it landed on -
    and the partial-coverage marker is deliberately left out of the test for
    the reason given above the patterns. The scan is bounded to the
    disposition's own paragraph, to the basis cell and to a line explicitly
    labelled Weighing or Review, so that a finding using the same words in its
    own prose is not read as the generator's statement about the disposition.
    """
    lines = md_lines(body)
    scanned = [basis or ""]
    found = labelled(lines, "Disposition")
    if found:
        first = found[0][0]
        for i, line in enumerate(lines[first:], first):
            # The line itself is read before the scan is allowed to stop on
            # it, because the note can arrive wearing anything - a heading, a
            # bullet, a numbered line - and stopping at the decoration would
            # step over the very statement being looked for. The scan ends at
            # the next thing the section announces, which is where the
            # disposition stops being what the report is talking about.
            scanned.append(line["text"])
            if i > first and (allegation_head(line) is not None
                              or (_opens_label(line) and not line_label(line)[1])):
                break
    for name in ("Weighing", "Review"):
        scanned += [value for _, value in labelled(lines, name)]

    blob = " ".join(scanned)
    counts = WEIGHED_COUNT.search(blob)
    weighed = int(counts.group(1)) if counts else None
    planned = int(counts.group(2)) if counts else None
    marker = UNSETTLED.search(blob)
    why = ""
    if marker:
        why = f"the section is marked {marker.group(0).lower()}"
    elif weighed == 0 and planned:
        why = f"0 of {planned} element(s) weighed"
    elif weighed == 0 and counts:
        why = "no element was weighed"
    return {"stated": bool(marker) or counts is not None,
            "weighed": weighed, "planned": planned,
            "default": bool(why), "why": why}


def _stated_disposition(block: dict) -> str:
    """The label written on this section's disposition line, undecorated.

    Read through the shared parser rather than against one spelling of the
    line: "**Disposition:** **Not substantiated**" puts bold on both sides of
    the colon and a pattern ending at the bold read the label as empty, which
    scores a stated disposition as a missing one.
    """
    found = labelled(md_lines(block["body"]), "Disposition")
    return found[0][1].strip() if found else ""


def score_report(key: dict, text: str) -> dict:
    """Disposition accuracy, allegation count, and summary/findings agreement.

    Accuracy here is not string equality against the key. A disposition
    produced by a procedural default - the element table could not be read, so
    the burden fell back on the allegation - is a statement about the
    generation rather than a finding about the evidence, and the report says
    as much beside it. Crediting it would mean a total collapse of the element
    pass scores as the right answer whenever the label it falls back to
    happens to be the label the key expects, which is the one failure this
    harness cannot see after the fact because from outside it looks like
    success. So a defaulted label is refused credit and named separately -
    and it is refused whichever verdict it landed on, since the test reads how
    the label was reached and never which label it is.

    Where the report states no weighing basis at all there is nothing to read
    and the labels are graded on their own, which is said in the output rather
    than passed over: a report that states a basis for some allegations and
    not others is a different matter, and the ones without it fail.
    """
    out: dict = {}
    blocks = allegation_blocks(text)
    findings = {n: _stated_disposition(blocks[n]) for n in sorted(blocks)}
    out["allegation_count"] = len(blocks)
    out["expected_count"] = len(key["allegations"])

    rows_by_number = summary_rows(text)
    want = [a["expected_disposition"] for a in key["allegations"]]
    weighings = {i: weighing(blocks[i]["body"] if i in blocks else "",
                             rows_by_number.get(i, {}).get("basis", ""))
                 for i in range(1, len(want) + 1)}
    anyone_stated = any(w["stated"] for w in weighings.values())

    credited, mislabelled, defaulted, unverified = 0, [], [], []
    for i, expected in enumerate(want, 1):
        found = findings.get(i, "")
        # Substring matching read "Not substantiated" as a correct
        # SUBSTANTIATED - the one comparison in this file that accepted the
        # exact opposite of the answer. Compare whole labels.
        if flat(found) != flat(expected):
            mislabelled.append(f"A{i}: key {expected}, report "
                               f"{found or '(none stated)'}")
            continue
        weighed = weighings[i]
        if weighed["default"]:
            defaulted.append(f"A{i}: {found} was not weighed - {weighed['why']}")
            continue
        if not weighed["stated"]:
            unverified.append(f"A{i}")
        credited += 1
    out["dispositions"] = f"{credited}/{len(want)}"
    out["mislabelled"] = mislabelled
    out["not_weighed"] = defaulted
    out["weighing_basis"] = (
        "stated per allegation" if anyone_stated and not unverified else
        "the report states none - labels graded on their own"
        if not anyone_stated else f"missing on {', '.join(unverified)}")

    # Summary must not contradict the finding blocks. A row whose allegation
    # text is long enough to wrap puts its verdict on the following line, and
    # a single-line pattern then matched no rows at all - so the check
    # reported the summary consistent on the strength of having read nothing.
    # md_lines joins a wrapped row back together before the columns are read.
    #
    # Agreement is a prefix relation, not a substring one. "substantiated"
    # sits inside "not substantiated", so the substring test read the exact
    # opposite verdict as agreement - the same hole the disposition
    # comparison above had. And "" is a substring of everything, so an
    # allegation with no finding block at all scored as agreeing with
    # whatever the table claimed about it.
    def agrees(found: str, verdict: str) -> bool:
        a, b = flat(found), flat(verdict)
        return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))
    mismatches = [n for n, row in sorted(rows_by_number.items())
                  if not agrees(findings.get(n, ""), row["verdict"])]
    out["summary_consistent"] = not mismatches
    out["summary_mismatch"] = [f"A{n}" for n in mismatches]
    out["conflicts_reported"] = sum(
        len(numbered_items(subsection(b["body"], "Conflicts")))
        for b in blocks.values())
    out["conflicts_expected"] = len(key.get("expected_conflicts", []))
    out["pass"] = (out["allegation_count"] == out["expected_count"]
                   and credited == len(want)
                   and (not unverified or not anyone_stated)
                   and out["summary_consistent"])
    return out

def score_corpus(key: dict, text: str, docs: list[dict]) -> dict:
    """Every ingested document has to show up in the report.

    A run that transcribed seven documents and then wrote its findings out of
    four dropped three documents' worth of evidence and still reported
    success. Only a citation counts: a filename in the persons list says the
    file exists, not that anything in it was read.
    """
    finished = [d for d in docs if (d.get("status") or "done") == "done"]
    pending = sorted(str(d.get("doc_id")) for d in docs
                     if (d.get("status") or "done") != "done")
    stems = doc_stems(finished, key)
    cites = [norm(c) for c in CITATION.findall(text)]
    cited = sorted(i for i, s in stems.items() if s and any(s in c for c in cites))
    uncited = sorted(set(stems) - set(cited))
    need = threshold(key, "corpus_completeness", default=len(stems))
    return {"cited": f"{len(cited)}/{len(stems)}", "needed": need,
            "never_cited": uncited, "unfinished": pending,
            "pass": bool(stems) and len(cited) >= need and not pending}


def score_citation_integrity(key: dict, text: str, docs: list[dict]) -> dict:
    """A finding's source has to be a document.

    Citing the allegation is the allegation proving itself: the claim under
    investigation is offered as the evidence for it, and no document was
    consulted at all. Any other bracket naming no ingested document is a
    source that does not exist.
    """
    stems = [s for s in doc_stems(docs, key).values() if s]
    blocks = allegation_blocks(text)
    circular, unresolved, uncited = [], [], []
    total = 0
    for n in sorted(blocks):
        own = {w for w in norm(blocks[n]["title"]).split() if len(w) > 4}
        for item in numbered_items(subsection(blocks[n]["body"], "Findings")):
            marks = CITATION.findall(item)
            if not marks:
                uncited.append(f"A{n}: {item[:44]}")
            for cite in marks:
                total += 1
                c = norm(cite)
                if any(s in c for s in stems):
                    continue
                # "Allegation" is the report template's own heading word, so a
                # bracket carrying it names a section of this document rather
                # than a source. The echo test catches the same move made by
                # quoting the allegation's wording instead of its number.
                if re.search(r"(?i)\balleg", cite) or sum(w in c for w in own) >= 3:
                    circular.append(f"A{n}: [{cite[:44]}]")
                else:
                    unresolved.append(f"A{n}: [{cite[:44]}]")
    good = total - len(circular) - len(unresolved)
    need = threshold(key, "citation_integrity", default=total)
    return {"citations": f"{good}/{total}", "needed": need,
            "circular": circular, "unresolved": unresolved,
            "findings_with_no_citation": uncited,
            "pass": total > 0 and good >= need}


def _ref_from_triples(item: str, tagged: list[dict]) -> set[int] | None:
    """Which allegation the extractor filed the quote behind this finding under."""
    shown = [q for q in (norm(x) for x in QUOTED.findall(item))
             if len(q.split()) >= 4]
    if not shown:
        return None
    hits: set[int] = set()
    for f in tagged:
        quote = norm(f.get("quote") or "")
        if quote and any(s in quote or quote in s for s in shown):
            hits |= {int(x) for x in re.findall(r"\d+", str(f["allegation_ref"]))}
    return hits or None


# A finding paraphrases the fact behind it rather than copying it, so the bar
# is a share of the gold fact's distinctive words rather than all of them.
# Both halves matter: a share alone lets a three-word fact match on one word,
# and a count alone lets a long fact match on a third of a sentence about
# something else. Below either, the finding is left untraced instead of
# guessed at. Both are defaults: a corpus whose findings paraphrase more
# loosely than this one's traces nothing at these numbers, and a check that
# measures nothing is not measuring the pipeline, so a key may lower them
# under scoring.tuning.
MIN_TRACE_SHARE = 0.33
MIN_TRACE_WORDS = 3


def _ref_from_gold(item: str, gold: list[dict],
                   min_words: int = MIN_TRACE_WORDS,
                   min_share: float = MIN_TRACE_SHARE) -> set[int] | None:
    """Which allegations the key says this finding's fact bears on.

    Where two gold facts fit the sentence equally well the sentence has not
    identified either of them, so the allegations of every equally good
    candidate are pooled: flagging on a coin toss would invent a misrouting
    that is not there.
    """
    blob = norm(item)
    item_words = [w for w in blob.split() if len(w) > 4]
    scored = []
    for g in gold:
        words = [w for w in norm(g["text"]).split() if len(w) > 4]
        if len(words) < min_words:
            continue
        hit = sum(w in blob for w in words)
        # Share is measured against the shorter of the two texts. Dividing by
        # the gold fact's own length penalises a finding for how verbosely the
        # key happened to write the fact: a finding recognising four
        # distinctive words of a sixteen-word fact scores 0.25 and is dropped,
        # while a shorter, wrong fact recognised by the same four words scores
        # 0.33 and wins. That is a bias toward short gold facts, and it flagged
        # a correctly-placed finding as misrouted.
        span = min(len(words), max(1, len(item_words)))
        if hit >= min_words and hit / span >= min_share:
            scored.append((hit / span, hit, g))
    if not scored:
        return None
    best = max(s for s, _, _ in scored)
    best_hits = max(h for _, h, _ in scored)
    pooled: set[int] = set()
    for share, hit, g in scored:
        # Equal on share, or equal on how many of the sentence's distinctive
        # words it accounts for. Share alone favours the shorter gold fact,
        # and two facts recognised by the same words are equally identified by
        # the sentence - the difference in share comes from the gold fact's own
        # length, which the finding had no say in. Preferring one of them on
        # that basis invented a misrouting where the finding matched the right
        # fact and a shorter wrong one equally well.
        if share >= best - 0.05 or hit >= best_hits:
            pooled |= {int(x) for x in re.findall(r"\d+", " ".join(g["allegations"]))}
    return pooled or None


def _page_carries(item: str, tagged: list[dict], number: int) -> bool:
    """Does a page this finding cites hold assertions tagged to this allegation?"""
    for doc, page in CITE_PAGE.findall(item):
        stem = norm(doc).rsplit(".", 1)[0]
        for f in tagged:
            if str(f.get("page_num")) != page:
                continue
            fid = norm(str(f.get("doc_id") or ""))
            if not (fid.startswith(stem) or stem.startswith(fid)):
                continue
            if number in {int(x) for x
                          in re.findall(r"\d+", str(f["allegation_ref"]))}:
                return True
    return False


CITE_PAGE = re.compile(r"\[([^\]\s,]+)[^\]]*?p{1,2}\.?\s*(\d+)[^\]]*\]", re.I)


def score_quote_routing(key: dict, text: str, facts: list[dict]) -> dict:
    """A finding under allegation N must rest on evidence bearing on N.

    Allegations are answered one at a time exactly so that one allegation's
    evidence cannot drift into another's answer: an admission given in reply
    to a question about the travel card proves nothing about the moves. Where
    triples carry allegation_ref, that column settles it. The column is
    nullable and older databases do not have it at all, so the fallback is the
    key's own per-fact allegation tags - the same defect read off the answer
    key instead of off the extractor.
    """
    gold = [g for g in key.get("gold_facts", [])
            if g.get("allegations") and g.get("text")]
    tagged = [f for f in facts if str(f.get("allegation_ref") or "").strip()]
    min_words = int(tuning(key, "min_trace_words", MIN_TRACE_WORDS))
    min_share = tuning(key, "min_trace_share", MIN_TRACE_SHARE)
    blocks = allegation_blocks(text)
    misrouted, checked = [], 0
    for n in sorted(blocks):
        for item in numbered_items(subsection(blocks[n]["body"], "Findings")):
            belongs = _ref_from_triples(item, tagged)
            if belongs is None:
                belongs = _ref_from_gold(item, gold, min_words, min_share)
            if not belongs:
                continue
            checked += 1
            # The extractor's own tagging vetoes a lexical guess. Where the
            # page this finding cites carries assertions tagged to THIS
            # allegation, the material is available here whatever gold fact the
            # wording happened to resemble - and the wording resembles the
            # wrong one often enough to matter: a finding about a credit card
            # traced to a fact about rank pressure on the strength of "order"
            # appearing inside "orders".
            if _page_carries(item, tagged, n):
                continue
            if n not in belongs:
                misrouted.append(f"A{n} rests on {sorted(belongs)} evidence: "
                                 f"{item[:52]}")
    need = threshold(key, "quote_routing", default=checked)
    return {"findings_traced": checked, "needed": need,
            "traced_by": "allegation_ref" if tagged else "key gold_facts tags",
            "misrouted": misrouted,
            "note": "" if checked else "no finding matched a fact this key "
                                       "tags by allegation - nothing measured",
            "pass": checked > 0 and checked - len(misrouted) >= need}


def score_conflict_opposition(key: dict, text: str) -> dict:
    """A conflict is two accounts that cannot both be true.

    Two witnesses recalling the same sentence the same way corroborate each
    other. Written up as a conflict that manufactures a dispute the evidence
    does not contain, and the adjudication underneath it is reasoning about
    nothing.
    """
    containment = tuning(key, "oppose_containment", OPPOSE_CONTAINMENT)
    blocks = allegation_blocks(text)
    reported, unscorable, agreements = 0, 0, []
    for n in sorted(blocks):
        for item in numbered_items(subsection(blocks[n]["body"], "Conflicts")):
            reported += 1
            sides = conflict_positions(item)
            if len(sides) < 2:
                unscorable += 1       # an observation limit has only one side
                continue
            if not any(opposes(a, b, containment)
                       for a, b in combinations(sides, 2)):
                agreements.append(f"A{n}: {item[:64]}")
    scored = reported - unscorable
    need = threshold(key, "conflict_precision", default=scored)
    # A report that surfaced no conflict at all has not shown precision, it has
    # skipped the question, and every other pass condition here is vacuously
    # true of an empty section. Only the key knows whether there was anything
    # to find, so an empty section passes only where the key expects none.
    measured = reported > 0 or not key.get("expected_conflicts")
    return {"conflicts_reported": reported, "adjudicable": scored,
            "needed": need, "single_sided": unscorable,
            "not_opposing": agreements,
            "pass": measured and scored - len(agreements) >= need}


def score_disposition_enum(key: dict, text: str) -> dict:
    """The disposition line is an enum and the key names its members.

    "Insufficient evidence" is a defensible thing for an investigator to think
    and an inadmissible thing to write on this line: the appointing authority
    is owed one of two answers per allegation, and a third label leaves the
    allegation open while reading as though it had been decided. The labels
    come out of the key so this does not become a list of the words one case
    happened to use.
    """
    labels = {flat(l) for l in key.get("meta", {}).get("disposition_labels", [])}
    blocks = allegation_blocks(text)
    if not labels:
        # A key that names no labels has not stated what the enum is, so there
        # is nothing here to be right or wrong about. Reporting FAIL on a
        # perfectly correct report because the key omitted an optional field
        # grades the key rather than the pipeline.
        return {"note": "the key names no disposition labels - not measured",
                "stated": {n: _stated_disposition(blocks[n]) for n in sorted(blocks)},
                "pass": True}
    stated, off_enum = {}, []
    for n in sorted(blocks):
        got = _stated_disposition(blocks[n])
        stated[n] = got
        if flat(got) not in labels:
            off_enum.append(f"A{n}: {got or '(none stated)'}")
    want = len(key.get("allegations", [])) or len(stated)
    # Its own category, not disposition_accuracy's. Borrowing that bar meant
    # tightening how many verdicts had to be RIGHT silently tightened how many
    # had to be spelled from the enum, which are different questions.
    need = threshold(key, "disposition_enum", default=want)
    return {"labels": sorted(labels), "stated": stated, "off_enum": off_enum,
            "allegations": f"{len(stated)}/{want}",
            "pass": len(stated) == want and len(stated) - len(off_enum) >= need}


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--report", help="a generated report .md to grade as well")
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    facts = rows("SELECT * FROM triples")
    try:
        docs = rows("SELECT doc_id, filename, status FROM documents")
    except Exception:
        docs = []          # an older state.db; the key's document list stands in
    try:
        ents = rows("SELECT entity_id, canonical_name, merged_into FROM entities")
    except Exception:
        ents = []
    print(f"case {key['meta']['case_id']}: {len(facts)} extracted fact(s)\n")

    sections = [
        ("entity_resolution", score_entities(key, facts)),
        ("negation_accuracy", score_negation(key, facts)),
        ("date_accuracy", score_dates(key, facts)),
        ("sourcing_discipline", score_sourcing(key, facts)),
        ("fact_recall", score_facts(key, facts)),
        ("merge_split", score_merges(key, facts)),
        ("merge_integrity", score_merge_graph(key, ents)),
        ("assertion_dedup", score_assertion_dedup(key, facts)),
        ("date_coverage", score_date_coverage(key, facts)),
        ("numeric_fidelity", score_numeric(key, facts)),
    ]
    if args.report:
        report_text = Path(args.report).read_text()
        sections.append(("corpus_completeness", score_corpus(key, report_text, docs)))
        sections.append(("conflict_recall", score_conflicts(key, report_text)))
        sections.append(("conflict_precision",
                         score_conflict_opposition(key, report_text)))
        sections.append(("quote_routing",
                         score_quote_routing(key, report_text, facts)))
        sections.append(("citation_integrity",
                         score_citation_integrity(key, report_text, docs)))
        sections.append(("disposition_enum",
                         score_disposition_enum(key, report_text)))
        sections.append(("exculpatory_recall", score_exculpatory(key, report_text)))
        sections.append(("credibility_handling", score_credibility(key, report_text)))
        sections.append(("report", score_report(key, report_text)))

    failed = 0
    for name, result in sections:
        ok = result.pop("pass", False)
        if not ok:
            failed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        for k, v in result.items():
            if isinstance(v, list) and len(v) > 3:
                v = v[:3] + [f"... {len(v) - 3} more"]
            print(f"         {k}: {v}")
    print(f"\n  {len(sections) - failed}/{len(sections)} categories passed")
    return failed



# ---------------------------------------------------------------------------
# Negative controls for the scorer itself.
#
# Every check in this file has at some point reported a pass it had not earned:
# a sourcing check that matched the wrong document, a numeric check blind to
# amounts written as words, a timeline check that let one extracted event
# satisfy four distinct gold entries. A check that cannot fail is not
# measuring anything. Each control below feeds a known defect to one check and
# asserts it fails.
# ---------------------------------------------------------------------------

def _finding_block(n: int, disposition: str, findings: str,
                   conflicts: str = "") -> str:
    """The smallest fragment of a report the report-side checks will parse."""
    return (f"#### Allegation {n}: the allegation under investigation\n\n"
            f"**Disposition:** {disposition}\n\n"
            f"**Findings**\n\n{findings}\n\n"
            f"**Conflicts in the evidence**\n\n"
            f"{conflicts or 'None identified.'}\n")


# What the generator writes beside a verdict, spelled the way it spells it,
# because the point of these controls is that the scorer reads the report as
# written rather than as the scorer would have written it. Three cases have to
# stay distinguishable: a table fully weighed, a table partly read that still
# entailed a verdict, and a table that entailed nothing so the verdict was
# defaulted. Only the last is refused credit, and only the note names it -
# a review marker sits on the middle case too, which is why the marker on its
# own cannot be the test.
_WEIGHED_BASIS = "5 of 5 element(s) weighed, 3 cited statement(s)"
_REVIEW_BASIS = ("**REVIEW REQUIRED** - element E5 was never weighed; "
                 "4 of 5 element(s) weighed, 3 cited statement(s)")
_PARTIAL_BASIS = ("**REVIEW REQUIRED** - element E5 was never weighed, so the "
                  "disposition rests on part of the element table only; "
                  "4 of 5 element(s) weighed, 3 cited statement(s)")
_UNSETTLED_NOTE = ("DISPOSITION UNSETTLED - OPERATOR REVIEW REQUIRED. Element E5 "
                   "was never weighed, so the element table entails nothing and "
                   "the label written in the section has not been accepted.")


def _graded_report(disposition: str, basis: str, note: str = "") -> str:
    """A finding block with the summary row the generator writes beside it.

    The basis cell is the point of the fixture. It sits after the verdict cell
    and carries decoration of its own, which is what made the verdict
    unreadable while the verdict was taken to be the row's last bold span.
    """
    return ("| # | Allegation | Disposition | Basis |\n|---|---|---|---|\n"
            f"| 1 | an allegation | **{disposition}** | {basis} |\n\n"
            f"#### Allegation 1: an allegation\n\n"
            f"**Disposition:** {disposition}\n\n"
            + (f"*{note}*\n\n" if note else "")
            + "**Findings**\n\n1. x [d.pdf p.1].\n")


def _no_basis_report() -> dict:
    """score_report over a summary table that records no weighing basis."""
    return score_report(
        {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
        "| # | Allegation | Disposition |\n|---|---|---|\n"
        "| 1 | an allegation | **Not substantiated** |\n\n" +
        _finding_block(1, "Not substantiated", "1. x [d.pdf p.1]."))


_TWO_DOCS = [{"doc_id": "d1", "filename": "DOC_A_Interview_Alpha__aaaaaaaa.pdf",
              "status": "done"},
             {"doc_id": "d2", "filename": "DOC_B_Interview_Bravo__bbbbbbbb.pdf",
              "status": "done"}]


SCORER_CONTROLS = [
    ("timeline rejects the right event on the wrong date",
     lambda: not score_dates(
         {"timeline_gold": [{"date": "2031-03-04", "precision": "day",
                             "event": "First warehouse key handover"}],
          "scoring": {"date_accuracy": {"pass_threshold": ">=1/1"}}},
         [{"quote": "key handover at the warehouse", "predicate": "made",
           "object_name": "handover", "subject_name": "Alpha",
           "event_date": "2031-05-09", "event_date_basis": "stated"}])["pass"]),

    ("timeline refuses to let one event satisfy two ordinals",
     lambda: score_dates(
         {"timeline_gold": [
             {"date": "2031-03-04", "precision": "day",
              "event": "First warehouse key handover"},
             {"date": "2031-05-09", "precision": "day",
              "event": "Second warehouse key handover"}],
          "scoring": {"date_accuracy": {"pass_threshold": ">=2/2"}}},
         [{"quote": "key handover at the warehouse", "predicate": "made",
           "object_name": "handover", "subject_name": "Alpha",
           "event_date": "2031-03-04", "event_date_basis": "stated"}]
     )["timeline_events"] == "1/2"),

    ("numeric accepts an amount written in words",
     lambda: score_numeric(
         {"gold_facts": [{"id": "F10", "text": "equipment charges of $206.45"}],
          "scoring": {"numeric_fidelity": {"metric": "F10 numeric fields exact"}}},
         [{"quote": "totaling 206 dollars and 45 cents", "object_name": ""}])["pass"]),

    ("numeric still fails on a changed digit",
     lambda: not score_numeric(
         {"gold_facts": [{"id": "F10", "text": "equipment charges of $206.45"}],
          "scoring": {"numeric_fidelity": {"metric": "F10 numeric fields exact"}}},
         [{"quote": "totaling 206 dollars and 46 cents", "object_name": ""}])["pass"]),

    ("sourcing fails when only the secondary document is cited",
     lambda: not score_sourcing(
         {"documents": [{"doc_id": "DOC_B", "file_match": "DOC_B_Interview_Bravo"},
                        {"doc_id": "DOC_C", "file_match": "DOC_C_Interview_Charlie"}],
          "gold_facts": [{"id": "F10",
                          "text": "eleven transactions totaling 906 dollars"}],
          "trap_inventory": {"sourcing": ["F10 primary source is DOC_B; "
                                          "restatements in DOC_C are secondary"]}},
         [{"quote": "eleven transactions totaling 906 dollars", "object_name": "",
           "doc_id": "DOC_C_Interview_Charlie_Subject"}])["pass"]),

    ("sourcing passes when the document of record is cited",
     lambda: score_sourcing(
         {"documents": [{"doc_id": "DOC_B", "file_match": "DOC_B_Interview_Bravo"}],
          "gold_facts": [{"id": "F10",
                          "text": "eleven transactions totaling 906 dollars"}],
          "trap_inventory": {"sourcing": ["F10 primary source is DOC_B"]}},
         [{"quote": "eleven transactions totaling 906 dollars", "object_name": "",
           "doc_id": "DOC_B_Interview_Bravo_GTC"}])["pass"]),


    ("corpus completeness fails on a document the report never cites",
     lambda: not score_corpus(
         {}, "1. A fact [DOC_A_Interview_Alpha__aaaaaaaa.pdf p.1].",
         _TWO_DOCS)["pass"]),

    ("corpus completeness passes when every document is cited",
     lambda: score_corpus(
         {}, "1. A [DOC_A_Interview_Alpha__aaaaaaaa.pdf p.1].\n"
             "2. B [DOC_B_Interview_Bravo__bbbbbbbb.pdf p.2].",
         _TWO_DOCS)["pass"]),

    ("corpus completeness honours a tolerance the key states",
     lambda: score_corpus(
         {"scoring": {"corpus_completeness": {"pass_threshold": ">=1/2"}}},
         "1. A fact [DOC_A_Interview_Alpha__aaaaaaaa.pdf p.1].",
         _TWO_DOCS)["pass"]),

    ("corpus completeness fails while a document is still unprocessed",
     lambda: not score_corpus(
         {}, "1. A [DOC_A_Interview_Alpha__aaaaaaaa.pdf p.1].",
         [_TWO_DOCS[0], dict(_TWO_DOCS[1], status="processing")])["pass"]),

    ("disposition enum rejects a third label",
     lambda: not score_disposition_enum(
         {"meta": {"disposition_labels": ["SUBSTANTIATED", "NOT_SUBSTANTIATED"]},
          "allegations": [{"id": "A1"}],
          "scoring": {"disposition_accuracy": {"pass_threshold": "1/1"}}},
         _finding_block(1, "Insufficient evidence", "1. x [d.pdf p.1]."))["pass"]),

    ("disposition enum accepts both labels the key names",
     lambda: score_disposition_enum(
         {"meta": {"disposition_labels": ["SUBSTANTIATED", "NOT_SUBSTANTIATED"]},
          "allegations": [{"id": "A1"}, {"id": "A2"}],
          "scoring": {"disposition_accuracy": {"pass_threshold": "2/2"}}},
         _finding_block(1, "Substantiated", "1. x [d.pdf p.1].") + "\n" +
         _finding_block(2, "Not substantiated", "1. y [d.pdf p.1]."))["pass"]),

    ("disposition enum fails when an allegation states none at all",
     lambda: not score_disposition_enum(
         {"meta": {"disposition_labels": ["SUBSTANTIATED", "NOT_SUBSTANTIATED"]},
          "allegations": [{"id": "A1"}],
          "scoring": {"disposition_accuracy": {"pass_threshold": "1/1"}}},
         "#### Allegation 1: x\n\n**Findings**\n\n1. y [d.pdf p.1].\n")["pass"]),

    ("quote routing fails when allegation 1 rests on allegation 2's evidence",
     lambda: not score_quote_routing(
         {"gold_facts": [{"id": "F10", "allegations": ["A2"],
                          "text": "eleven transactions totaling 906 dollars "
                                  "on the unit purchase card"}]},
         _finding_block(1, "Substantiated",
                        "1. The subject made eleven transactions totaling "
                        "906 dollars on the unit purchase card [d.pdf p.1]."),
         [])["pass"]),

    ("quote routing passes when the finding rests on its own allegation",
     lambda: score_quote_routing(
         {"gold_facts": [{"id": "F10", "allegations": ["A1"],
                          "text": "eleven transactions totaling 906 dollars "
                                  "on the unit purchase card"}]},
         _finding_block(1, "Substantiated",
                        "1. The subject made eleven transactions totaling "
                        "906 dollars on the unit purchase card [d.pdf p.1]."),
         [])["pass"]),

    ("quote routing falls back to the key when no triple carries allegation_ref",
     lambda: score_quote_routing(
         {"gold_facts": [{"id": "F10", "allegations": ["A1"],
                          "text": "eleven transactions totaling 906 dollars "
                                  "on the unit purchase card"}]},
         _finding_block(1, "Substantiated",
                        "1. The subject made eleven transactions totaling "
                        "906 dollars on the unit purchase card [d.pdf p.1]."),
         [{"quote": "some other quote entirely", "predicate": "said"}]
     )["traced_by"] == "key gold_facts tags"),

    ("quote routing reads allegation_ref ahead of the key where it is populated",
     lambda: not score_quote_routing(
         {"gold_facts": [{"id": "F10", "allegations": ["A1"],
                          "text": "the subject used the card for personal "
                                  "charges at all times"}]},
         _finding_block(1, "Substantiated",
                        '1. The subject said "I never used the card for '
                        'personal charges at all" [d.pdf p.1].'),
         [{"quote": "I never used the card for personal charges at all",
           "allegation_ref": "A2"}])["pass"]),

    ("conflict check fails on two sources saying the same thing",
     lambda: not score_conflict_opposition(
         {}, _finding_block(
             1, "Substantiated", "1. x [d.pdf p.1].",
             '1. The subject wrote that it "had nothing to do with work" '
             '[a.pdf p.1]. The witness recalled he said "this has nothing to '
             'do with work" [b.pdf p.1].'))["pass"]),

    ("conflict check accepts an account the other side denies",
     lambda: score_conflict_opposition(
         {}, _finding_block(
             1, "Substantiated", "1. x [d.pdf p.1].",
             '1. The witness said "the crew worked the whole shift on site" '
             '[a.pdf p.1]. The subject said "the crew never worked the shift '
             'on site" [b.pdf p.1].'))["pass"]),

    ("conflict check accepts accounts differing on a date",
     lambda: score_conflict_opposition(
         {}, _finding_block(
             1, "Substantiated", "1. x [d.pdf p.1].",
             '1. The complainant said the move was "on weekdays in October" '
             '[a.pdf p.1]. The witness said it was "on Saturday 9 March" '
             '[b.pdf p.1].'))["pass"]),

    ("conflict check leaves a one-sided observation limit unjudged",
     lambda: score_conflict_opposition(
         {}, _finding_block(
             1, "Substantiated", "1. x [d.pdf p.1].",
             "1. The witness stated she only knew of the delivery because "
             "a coworker told her about it the same day [a.pdf p.1]."
         ))["single_sided"] == 1),

    ("conflict precision honours a tolerance the key states",
     lambda: score_conflict_opposition(
         {"scoring": {"conflict_precision": {"pass_threshold": "1/2"}}},
         _finding_block(
             1, "Substantiated", "1. x [d.pdf p.1].",
             '1. The subject wrote that it "had nothing to do with work" '
             '[a.pdf p.1]. The witness recalled he said "this has nothing to '
             'do with work" [b.pdf p.1].\n'
             '2. The witness said "the crew worked the whole shift on site" '
             '[a.pdf p.1]. The subject said "the crew never worked the shift '
             'on site" [b.pdf p.1].'))["pass"]),

    ("conflict precision is not satisfied by a report that surfaces none",
     lambda: not score_conflict_opposition(
         {"expected_conflicts": [{"id": "C1"}]},
         _finding_block(1, "Substantiated", "1. x [d.pdf p.1].",
                        "None identified."))["pass"]),

    ("citation integrity fails on a finding citing the allegation itself",
     lambda: not score_citation_integrity(
         {}, _finding_block(1, "Substantiated",
                            "1. The subject sent an email to his flight chief "
                            "[Allegation 1]."),
         [_TWO_DOCS[0]])["pass"]),

    ("citation integrity fails on a source no ingested document answers to",
     lambda: not score_citation_integrity(
         {}, _finding_block(1, "Substantiated",
                            "1. The card was flagged by finance "
                            "[Finance system report p.4]."),
         [_TWO_DOCS[0]])["pass"]),

    ("citation integrity passes when every finding cites a document",
     lambda: score_citation_integrity(
         {}, _finding_block(1, "Substantiated",
                            "1. The subject sent an email "
                            "[DOC_A_Interview_Alpha__aaaaaaaa.pdf p.1]."),
         [_TWO_DOCS[0]])["pass"]),

    ("disposition accuracy is not satisfied by the opposite verdict",
     lambda: not score_report(
         {"allegations": [{"expected_disposition": "SUBSTANTIATED"}]},
         "#### Allegation 1: x\n\n**Disposition:** Not substantiated\n\n"
         "| 1 | x | **Not substantiated** |\n")["pass"]),

    ("disposition accuracy accepts the key's underscored spelling of a label",
     lambda: score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
         "#### Allegation 1: x\n\n**Disposition:** Not substantiated\n\n"
         "| 1 | x | **Not substantiated** |\n")["pass"]),

    ("summary consistency reads a table row that wraps onto a second line",
     lambda: not score_report(
         {"allegations": [{"expected_disposition": "SUBSTANTIATED"}]},
         "#### Allegation 1: x\n\n**Disposition:** Substantiated\n\n"
         "| # | Allegation | Disposition |\n|---|---|---|\n"
         "| 1 | That between 1 April and 31 July the subject improperly\n"
         "use | **Not substantiated** |\n")["summary_consistent"]),

    ("summary consistency is not satisfied by a missing finding block",
     lambda: not score_report(
         {"allegations": [{"expected_disposition": "SUBSTANTIATED"},
                          {"expected_disposition": "SUBSTANTIATED"}]},
         "#### Allegation 1: x\n\n**Disposition:** Substantiated\n\n"
         "| 1 | x | **Substantiated** |\n"
         "| 2 | y | **Substantiated** |\n")["summary_consistent"]),

    # ---- the verdict column, and how the verdict was reached --------------
    # The row is written verdict cell first, basis cell second, and the basis
    # carries decoration of its own. Taking the verdict as the row's last bold
    # span therefore read the basis instead on exactly the allegations whose
    # weighing went wrong, and reported a summary inconsistency against a
    # report whose summary agreed with its findings perfectly.
    ("the summary verdict is read from its column, not the row's last bold span",
     lambda: score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
         _graded_report("Not substantiated", _REVIEW_BASIS))["summary_consistent"]),

    ("a summary that does contradict its block is still caught beside that cell",
     lambda: not score_report(
         {"allegations": [{"expected_disposition": "SUBSTANTIATED"}]},
         _graded_report("Substantiated", _REVIEW_BASIS).replace(
             "**Disposition:** Substantiated",
             "**Disposition:** Not substantiated"))["summary_consistent"]),

    # Disposition accuracy has to read how the label was reached, not only
    # what it says. A label the generator defaulted to because it could not
    # weigh the elements is a statement about the generation, and crediting it
    # scores a collapse of the element pass as the right answer whenever the
    # default happens to land on the label the key expects. The next two
    # controls are the same fixture under opposite keys: the refusal has to be
    # blind to which verdict was defaulted to, or it is steering.
    ("a disposition the report says nobody weighed is not credited",
     lambda: not score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
         _graded_report("Not substantiated", _REVIEW_BASIS,
                        _UNSETTLED_NOTE))["pass"]),

    ("the same default is refused credit on the opposite verdict",
     lambda: not score_report(
         {"allegations": [{"expected_disposition": "SUBSTANTIATED"}]},
         _graded_report("Substantiated", _REVIEW_BASIS,
                        _UNSETTLED_NOTE))["pass"]),

    ("a weighed disposition is credited on either verdict",
     lambda: score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
         _graded_report("Not substantiated", _WEIGHED_BASIS))["pass"]
     and score_report(
         {"allegations": [{"expected_disposition": "SUBSTANTIATED"}]},
         _graded_report("Substantiated", _WEIGHED_BASIS))["pass"]),

    ("a verdict entailed on part of the element table is still credited",
     lambda: score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
         _graded_report("Not substantiated", _PARTIAL_BASIS))["pass"]
     and score_report(
         {"allegations": [{"expected_disposition": "SUBSTANTIATED"}]},
         _graded_report("Substantiated", _PARTIAL_BASIS))["pass"]),

    ("a verdict resting on no element weighed at all is a default too",
     lambda: not score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
         _graded_report("Not substantiated",
                        "0 of 5 element(s) weighed, 0 cited statement(s)"))["pass"]),

    # A report that records no weighing basis anywhere leaves nothing to read
    # about how its labels were reached, and inventing a failure out of that
    # silence would grade the generator's annotations rather than the
    # investigation. It is graded on the labels and the output says so, which
    # is the difference between a measurement and an assumption.
    ("a report stating no weighing basis is graded on its labels and says so",
     lambda: _no_basis_report()["pass"]
     and "states none" in _no_basis_report()["weighing_basis"]),

    ("a basis stated for one allegation and missing on another does not pass",
     lambda: not score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"},
                          {"expected_disposition": "NOT_SUBSTANTIATED"}]},
         "| # | Allegation | Disposition | Basis |\n|---|---|---|---|\n"
         f"| 1 | an allegation | **Not substantiated** | {_WEIGHED_BASIS} |\n"
         "| 2 | another allegation | **Not substantiated** |  |\n\n" +
         _finding_block(1, "Not substantiated", "1. x [d.pdf p.1].") + "\n" +
         _finding_block(2, "Not substantiated", "1. y [d.pdf p.1]."))["pass"]),

    # A rule that can only ever reach one disposition is steering however it
    # is spelled, so the test for a default is checked to name neither: it
    # reads the generator's own account of how much was weighed, and there is
    # no verdict word in it to prefer.
    ("the weighing test names no disposition, so it cannot prefer one",
     lambda: not re.search(r"(?i)substantiat|sustain|founded",
                           UNSETTLED.pattern + WEIGHED_COUNT.pattern)),

    ("an element table below the summary is not read as a summary row",
     lambda: score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
         "| # | Allegation | Disposition | Basis |\n|---|---|---|---|\n"
         f"| 1 | an allegation | **Not substantiated** | {_WEIGHED_BASIS} |\n\n"
         "#### Allegation 1: an allegation\n\n"
         "**Elements**\n\n| # | Element | Met |\n|---|---|---|\n"
         "| 1 | the first element | Yes |\n\n"
         "**Disposition:** Not substantiated\n\n"
         "**Findings**\n\n1. x [d.pdf p.1].\n")["pass"]),

    # ---- decoration, wherever the model puts it --------------------------
    # Each of these is a form that a regex written against one spelling of the
    # anchor read as nothing at all, and reading an anchor as nothing is how a
    # check comes to report a pass on a document it never parsed.
    ("a disposition is read with decoration on both sides of the colon",
     lambda: score_report(
         {"allegations": [{"expected_disposition": "NOT_SUBSTANTIATED"}]},
         _graded_report("Not substantiated", _WEIGHED_BASIS).replace(
             "**Disposition:** Not substantiated",
             "**Disposition:** **Not substantiated**"))["pass"]),

    ("a findings heading is read with a trailing colon inside its bold",
     lambda: len(numbered_items(subsection(
         "\n**Disposition:** Not substantiated\n\n**Findings:**\n\n"
         "1. x [d.pdf p.1].\n2. y [e.pdf p.2].\n\n**Gaps**\n\nNone.\n",
         "Findings"))) == 2),

    ("a findings list written as a pipe table is read as findings",
     lambda: len(numbered_items(subsection(
         "\n**Findings**\n\n| # | Finding |\n|---|---|\n"
         "| 1 | x [d.pdf p.1]. |\n| 2 | y [e.pdf p.2]. |\n", "Findings"))) == 2),

    ("an allegation heading is read in bold and as a bullet",
     lambda: sorted(allegation_blocks(
         "**Allegation 1: the first**\n\n**Disposition:** Substantiated\n\n"
         "- Allegation 2: the second\n\n**Disposition:** Not substantiated\n"
     )) == [1, 2]),

    ("a finding that mentions another allegation does not open a section",
     lambda: sorted(allegation_blocks(
         "#### Allegation 1: the first\n\n**Disposition:** Substantiated\n\n"
         "**Findings**\n\n1. Allegation 2 rests on the same email [d.pdf p.1].\n"
     )) == [1]),

    ("a conflicts heading is read with a trailing colon inside its bold",
     lambda: score_conflicts(
         {"expected_conflicts": [{"id": "C1", "positions": {
             "driver": "the shipment left the yard before midnight",
             "guard": "the shipment never left the yard that night"}}],
          "scoring": {"conflict_recall": {"pass_threshold": "1/1 incl C1"}}},
         "#### Allegation 1: x\n\n**Disposition:** Not substantiated\n\n"
         "**Findings**\n\n1. y [d.pdf p.1].\n\n"
         "**Conflicts in the evidence:**\n\n"
         '1. The driver said "the shipment left the yard before midnight" '
         '[a.pdf p.1]. The guard said "the shipment never left the yard that '
         'night" [b.pdf p.2].\n')["pass"]),

    ("exculpatory routing is not satisfied by a fact filed under another allegation",
     lambda: not score_exculpatory(
         {"gold_facts": [{"id": "F04", "text": "the transfers occurred at weekends "
                                               "while the section was released"}],
          "trap_inventory": {"exculpatory_routing": [
              "F04 is exculpatory on A1 and must appear in the A1 analysis"]},
          "scoring": {"exculpatory_recall": {"pass_threshold": "1/1"}}},
         _finding_block(1, "Substantiated", "1. x [d.pdf p.1].") + "\n" +
         _finding_block(2, "Substantiated",
                        "1. The transfers occurred at weekends while the "
                        "section was released [d.pdf p.1]."))["pass"]),

    ("exculpatory routing passes when the fact reaches the right allegation",
     lambda: score_exculpatory(
         {"gold_facts": [{"id": "F04", "text": "the transfers occurred at weekends "
                                               "while the section was released"}],
          "trap_inventory": {"exculpatory_routing": [
              "F04 is exculpatory on A1 and must appear in the A1 analysis"]},
          "scoring": {"exculpatory_recall": {"pass_threshold": "1/1"}}},
         _finding_block(1, "Substantiated",
                        "1. The transfers occurred at weekends while the "
                        "section was released [d.pdf p.1].") + "\n" +
         _finding_block(2, "Substantiated", "1. x [d.pdf p.1]."))["pass"]),

    # ---- merge graph integrity -------------------------------------------
    # score_merges asks whether a named alias was merged. Both of these
    # fixtures answer yes to that question while leaving resolution with
    # nowhere to land, which is why that check passed a graph containing a
    # two-node cycle.
    ("merge graph fails on two entities merged into each other",
     lambda: not score_merge_graph({}, [
         {"entity_id": "PERSON:alpha", "merged_into": "PERSON:nickname"},
         {"entity_id": "PERSON:nickname", "merged_into": "PERSON:alpha"}])["pass"]),

    ("merge graph fails on an entity merged into itself",
     lambda: not score_merge_graph({}, [
         {"entity_id": "PERSON:alpha", "merged_into": "PERSON:alpha"}])["pass"]),

    ("merge graph fails on a merge into an entity that does not exist",
     lambda: not score_merge_graph({}, [
         {"entity_id": "PERSON:alpha", "merged_into": "PERSON:absent"}])["pass"]),

    ("merge graph passes when every chain ends at an unmerged root",
     lambda: score_merge_graph({}, [
         {"entity_id": "PERSON:nickname", "merged_into": "PERSON:alias"},
         {"entity_id": "PERSON:alias", "merged_into": "PERSON:alpha"},
         {"entity_id": "PERSON:alpha", "merged_into": None},
         {"entity_id": "PERSON:bravo", "merged_into": ""}])["pass"]),

    ("merge graph is not satisfied by a graph holding no entities at all",
     lambda: not score_merge_graph({}, [])["pass"]),

    # ---- assertion de-duplication ----------------------------------------
    ("assertion dedup fails on one sentence extracted at four lengths",
     lambda: not score_assertion_dedup({}, [
         {"doc_id": "d1", "page_num": 1, "subject_name": "Alpha",
          "predicate": "stated", "quote": q}
         for q in ("It was false.",
                   "It was false. I panicked.",
                   "It was false. I panicked. I meant to fix it.",
                   "It was false. I panicked. I meant to fix it. I did not.")
     ])["pass"]),

    ("assertion dedup passes on two different statements by one person",
     lambda: score_assertion_dedup({}, [
         {"doc_id": "d1", "page_num": 1, "subject_name": "Alpha",
          "predicate": "stated", "quote": "The shipment left on time."},
         {"doc_id": "d1", "page_num": 1, "subject_name": "Alpha",
          "predicate": "stated", "quote": "Nobody signed the manifest."}])["pass"]),

    # Corroboration is not duplication: the same sentence reaching the record
    # through two documents is two sources for it, and deducting for that
    # would punish the pipeline for reading both.
    ("assertion dedup does not charge two documents for quoting one sentence",
     lambda: score_assertion_dedup({}, [
         {"doc_id": "d1", "page_num": 1, "subject_name": "Alpha",
          "predicate": "stated", "quote": "The shipment left on time."},
         {"doc_id": "d2", "page_num": 4, "subject_name": "Alpha",
          "predicate": "stated", "quote": "The shipment left on time."}])["pass"]),

    ("assertion dedup honours a tolerance the key states",
     lambda: score_assertion_dedup(
         {"scoring": {"tuning": {"max_duplicate_assertions": 1}}}, [
             {"doc_id": "d1", "page_num": 1, "subject_name": "Alpha",
              "predicate": "stated", "quote": "It was false."},
             {"doc_id": "d1", "page_num": 1, "subject_name": "Alpha",
              "predicate": "stated", "quote": "It was false. I panicked."}
         ])["pass"]),

    ("assertion dedup is not satisfied by an empty set of assertions",
     lambda: not score_assertion_dedup({}, [])["pass"]),

    # ---- date coverage ----------------------------------------------------
    ("date coverage fails when dated sentences arrive with no date attached",
     lambda: not score_date_coverage({}, [
         {"quote": "The crew stood down on 14 July 2031.", "event_date": ""},
         {"quote": "The manifest was signed on 15 July 2031.", "event_date": ""},
         {"quote": "The second run began in August 2031.",
          "event_date": "2031-08-01"}])["pass"]),

    ("date coverage passes when the sentences that name a time carry one",
     lambda: score_date_coverage({}, [
         {"quote": "The crew stood down on 14 July 2031.",
          "event_date": "2031-07-14"},
         {"quote": "The manifest was signed on 15 July 2031.",
          "event_date": "2031-07-15"},
         {"quote": "The section holds the standing duty roster.",
          "event_date": ""}])["pass"]),

    ("date coverage reads its bar from the key",
     lambda: score_date_coverage(
         {"scoring": {"date_coverage": {"pass_threshold": ">=1/3"}}}, [
             {"quote": "The crew stood down on 14 July 2031.", "event_date": ""},
             {"quote": "The manifest was signed on 15 July 2031.", "event_date": ""},
             {"quote": "The second run began in August 2031.",
              "event_date": "2031-08-01"}])["pass"]),

    ("date coverage does not charge an assertion whose text names no time",
     lambda: score_date_coverage({}, [
         {"quote": "The subject held the section NCOIC billet.", "event_date": ""},
         {"quote": "The crew stood down on 14 July 2031.",
          "event_date": "2031-07-14"}])["datable"] == "1/1"),

    ("date coverage does not read the modal verb may as a month",
     lambda: score_date_coverage({}, [
         {"quote": "The witness said he may have signed it.", "event_date": ""},
         {"quote": "The crew stood down on 14 May 2031.",
          "event_date": "2031-05-14"}])["datable"] == "1/1"),

    # ---- constants that belong to the key, not to this file ---------------
    ("disposition enum skips rather than fails when the key names no labels",
     lambda: score_disposition_enum(
         {"allegations": [{"id": "A1"}]},
         _finding_block(1, "Insufficient evidence",
                        "1. x [d.pdf p.1]."))["pass"]),

    ("disposition enum reads its own bar, not disposition accuracy's",
     lambda: score_disposition_enum(
         {"meta": {"disposition_labels": ["SUBSTANTIATED", "NOT_SUBSTANTIATED"]},
          "allegations": [{"id": "A1"}, {"id": "A2"}],
          "scoring": {"disposition_accuracy": {"pass_threshold": "2/2"},
                      "disposition_enum": {"pass_threshold": "1/2"}}},
         _finding_block(1, "Substantiated", "1. x [d.pdf p.1].") + "\n" +
         _finding_block(2, "Insufficient evidence", "1. y [d.pdf p.1]."))["pass"]),

    ("a tuning constant is read from the key, not from this file",
     lambda: tuning({"scoring": {"tuning": {"min_trace_share": 0.2}}},
                    "min_trace_share", default=0.33) == 0.2
     and tuning({}, "min_trace_share", default=0.33) == 0.33),

    ("conflict opposition reads its containment bar from the key",
     lambda: score_conflict_opposition(
         {"scoring": {"tuning": {"oppose_containment": 1.0}}},
         _finding_block(
             1, "Substantiated", "1. x [d.pdf p.1].",
             '1. The subject wrote that it "had nothing to do with work" '
             '[a.pdf p.1]. The witness recalled he said "this has nothing to '
             'do with work" [b.pdf p.1].'))["pass"]),

    ("a document stem drops a hash suffix joined by a single separator",
     lambda: doc_stems([{"doc_id": "d1",
                         "filename": "DOC_A_Interview_Alpha-9f3c1b7e.pdf"}], {})
     == {"d1": "doc a interview alpha"}),

    ("a document stem keeps a name that merely looks hexadecimal",
     lambda: doc_stems([{"doc_id": "d1",
                         "filename": "DOC_A_Statement_facade.pdf"}], {})
     == {"d1": "doc a statement facade"}),

    ("threshold is read from the key, not from this file",
     lambda: threshold({"scoring": {"date_accuracy":
                                    {"pass_threshold": ">=19/22"}}},
                       "date_accuracy", default=0) == 19),
]


SCORER_CONTROLS += [
    ("stem matching reaches an inflection of the same word",
     lambda: word_present("referral", "referred it to his flight chief")),

    ("stem matching does not reach a merely similar word",
     lambda: not word_present("attended", "he attempted the move")),

    ("stem matching needs a real stem, not two letters",
     lambda: not word_present("was", "wasp nest in the hangar")),

    ("three objects in one sentence are three facts, not a repetition",
     lambda: score_assertion_dedup(
         {}, [{"doc_id": "D", "page_num": 1, "subject_name": "IO",
               "predicate": "took custody of", "object_name": f"exhibit {n}",
               "event_date": "2026-08-28", "quote": "the IO took custody of "
               "exhibits 5, 6 and 8"} for n in (5, 6, 8)])["pass"]),

    ("the same object quoted at two lengths is still a repetition",
     lambda: not score_assertion_dedup(
         {}, [{"doc_id": "D", "page_num": 1, "subject_name": "IO",
               "predicate": "took custody of", "object_name": "exhibit 5",
               "event_date": "2026-08-28", "quote": q}
              for q in ("the IO took custody of exhibit 5",
                        "the IO took custody of exhibit 5, which was the "
                        "email thread")])["pass"]),

    ("a dated series is not a repetition",
     lambda: score_assertion_dedup(
         {}, [{"doc_id": "D", "page_num": 1, "subject_name": "Osei",
               "predicate": "recorded", "object_name": "a cash advance",
               "event_date": d, "quote": "advances dated 6 June, 20 June, "
               "3 July and 17 July"}
              for d in ("2026-06-06", "2026-06-20", "2026-07-03")])["pass"]),
]


SCORER_CONTROLS += [
    ("hearsay counts only when its weight is addressed",
     lambda: not score_credibility(
         {}, "The witness credibility and motive are discussed. He heard about "
             "the move from a colleague.")["pass"]),

    ("hearsay noticed and discounted passes however it is phrased",
     lambda: score_credibility(
         {}, "Complainant motive is discussed. Nothing firsthand; he heard "
             "about it after the fact, which limits the weight of the "
             "account.")["pass"]),

    ("discounting language alone is not hearsay handling",
     lambda: not score_credibility(
         {}, "The evidence carries little weight and the limits are noted.")
         ["pass"]),
]


SCORER_CONTROLS += [
    ("a finding matching two gold facts equally well is not misrouted",
     lambda: _ref_from_gold(
         "Brandt stated he knew the rules and used the card for personal "
         "expenses because he was in a financial hole",
         [{"id": "F13", "allegations": ["A2"],
           "text": "Brandt admits charges are personal, knew the rules, used "
                   "the card because he was in a financial hole"},
          {"id": "F15", "allegations": ["A3"],
           "text": "Brandt stated he knew the rules regarding official "
                   "expenses"}], 3, 0.2) == {2, 3}),

    ("a finding matching one gold fact traces only there",
     lambda: _ref_from_gold(
         "Brandt made four cash advances at a casino in Henderson Nevada",
         [{"id": "F10", "allegations": ["A2"],
           "text": "four cash advances at a casino in Henderson Nevada"},
          {"id": "F20", "allegations": ["A1"],
           "text": "subordinates moved household goods on a Saturday"}],
         3, 0.2) == {2}),
]


SCORER_CONTROLS += [
    ("a verbosely written gold fact is not penalised for its length",
     lambda: 2 in (_ref_from_gold(
         "Brandt knew the rules and used the card for personal expenses "
         "because he was in a financial hole",
         [{"id": "F13", "allegations": ["A2"],
           "text": "Brandt admits all nine charges are his and personal with "
                   "no official nexus, knew the rules, used the card because "
                   "he was in a financial hole and it was the only one with "
                   "room"}], 3, 0.33) or set())),
]


SCORER_CONTROLS += [
    ("a conflict stated without positions can still be credited",
     lambda: score_conflicts(
         {"expected_conflicts": [{"id": "C5", "positions": {},
           "expected_resolution": "the hearsay statements carry no independent "
                                  "weight and the allegation rests on certified "
                                  "records and the subject admission"}],
          "scoring": {"conflict_recall": {"pass_threshold": ">=1/1 incl C5"}}},
         "**Conflicts in the evidence**\n1. TYPE: observation-limit\n"
         "The hearsay statements carry no independent weight; the allegation "
         "rests on the certified records and the subject admission."
     )["detected"] == ["C5"]),

    ("a four-letter word in a position is not discarded",
     lambda: score_conflicts(
         {"expected_conflicts": [{"id": "C1", "positions": {
             "a": "duty days in June", "b": "Saturdays off duty"}}],
          "scoring": {"conflict_recall": {"pass_threshold": ">=1/1 incl C1"}}},
         "**Conflicts in the evidence**\n1. TYPE: contradiction\n"
         "The complaint puts the moves on duty days in June; the witnesses put "
         "them on Saturdays, off duty."
     )["detected"] == ["C1"]),
]


SCORER_CONTROLS += [
    ("an account named as carrying no independent weight counts as hearsay",
     lambda: score_credibility(
         {}, "Complainant motive is discussed. Those statements carry no "
             "independent weight and the allegation rests on the certified "
             "records.")["pass"]),

    ("weight language with no account identified still fails",
     lambda: not score_credibility(
         {}, "Motive is discussed. The evidence carries weight and the limits "
             "are noted.")["pass"]),
]


SCORER_CONTROLS += [
    ("the extractor's page tagging vetoes a lexical mis-trace",
     lambda: _page_carries(
         "Brandt used the card anyway [D2_11_Interview.pdf p.1].",
         [{"doc_id": "D2_11_Interview", "page_num": 1, "allegation_ref": "2"}],
         2)),

    ("a page tagged only elsewhere does not veto",
     lambda: not _page_carries(
         "Brandt sent the email [D2_11_Interview.pdf p.2].",
         [{"doc_id": "D2_11_Interview", "page_num": 2, "allegation_ref": "3"}],
         2)),
]


def self_test() -> int:
    print("negative controls - every scorer check must fail on its own defect\n")
    passed = 0
    for name, probe in SCORER_CONTROLS:
        try:
            ok = bool(probe())
        except Exception as exc:                       # a crash is also a fail
            ok, name = False, f"{name} [raised {type(exc).__name__}: {exc}]"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        passed += ok
    print(f"\n  {passed}/{len(SCORER_CONTROLS)} controls fired")
    return 0 if passed == len(SCORER_CONTROLS) else 1


if __name__ == "__main__":
    sys.exit(main())
