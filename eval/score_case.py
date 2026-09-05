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
    conflict_recall     conflicts detected (report mode only)
    disposition_accuracy per allegation, plus summary/findings agreement

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
    """Each negation trap must appear with its negation intact, and its positive
    twin must not appear at all."""
    traps = key.get("trap_inventory", {}).get("negation", [])
    NEG = re.compile(r"\b(never|not|no |zero|failed|denied|refused|without)\b", re.I)
    results = []
    for trap in traps:
        # A trap is described in prose; match on its distinctive content words.
        words = [w for w in norm(trap).split() if len(w) > 4][:4]
        related = [f for f in facts
                   if sum(w in norm(f["predicate"] + " " + f["object_name"] +
                                    " " + f["quote"]) for w in words) >= 2]
        if not related:
            results.append({"trap": trap[:60], "status": "not found"})
            continue
        negated = any(NEG.search(f["predicate"]) or NEG.search(f["object_name"])
                      or NEG.search(f["quote"]) for f in related)
        results.append({"trap": trap[:60], "status": "kept" if negated else "POLARITY LOST"})
    lost = [r for r in results if r["status"] == "POLARITY LOST"]
    return {"traps": results, "lost": len(lost), "pass": not lost}


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
    return {"precision_upgrades": upgrades[:10], "count": len(upgrades),
            "pass": not upgrades}


def score_sourcing(key: dict, facts: list[dict]) -> dict:
    """A documentary figure must be cited to the document of record, not to a
    person restating it."""
    docs = {d["doc_id"]: d["file_match"] for d in key["documents"]}
    results = []
    for trap in key.get("trap_inventory", {}).get("sourcing", []):
        ids = re.findall(r"\b(\d{2})\b", trap)
        if len(ids) < 2:
            continue
        primary = ids[-1]
        words = [w for w in norm(trap).split() if len(w) > 5][:3]
        related = [f for f in facts
                   if sum(w in norm(f["quote"]) for w in words) >= 1]
        cited = {f["doc_id"] for f in related}
        want = docs.get(primary, "")
        ok = any(want and want.lower()[:12] in c.lower() for c in cited)
        results.append({"trap": trap[:60], "cited_primary": ok,
                        "docs": sorted(c[:26] for c in cited)[:3]})
    passed = sum(1 for r in results if r["cited_primary"])
    return {"checks": results, "pass": bool(results) and passed == len(results)}


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
    return {"recalled": len(recalled), "total": len(key["gold_facts"]),
            "missed": missed, "pass": len(recalled) >= 22}


def score_report(key: dict, text: str) -> dict:
    """Disposition accuracy, allegation count, and summary/findings agreement."""
    out: dict = {}
    blocks = re.findall(r"#### Allegation (\d+):(.*?)\*\*Disposition:\*\*\s*(.+)",
                        text, re.S)
    out["allegation_count"] = len(blocks)
    out["expected_count"] = len(key["allegations"])
    findings = {int(n): d.strip().split("\n")[0] for n, _, d in blocks}

    want = [a["expected_disposition"] for a in key["allegations"]]
    correct = 0
    for i, expected in enumerate(want, 1):
        got = findings.get(i, "")
        if expected.lower().split()[0] in got.lower():
            correct += 1
    out["dispositions"] = f"{correct}/{len(want)}"

    # Summary must not contradict the finding blocks.
    table = re.findall(r"^\|\s*(\d+)\s*\|.*?\|\s*\*\*(.+?)\*\*\s*\|", text, re.M)
    mismatches = [n for n, verdict in table
                  if findings.get(int(n), "").lower()[:12] not in verdict.lower()]
    out["summary_consistent"] = not mismatches
    conflicts = len(re.findall(r"(?i)^\s*\d+\.\s", 
                    "\n".join(re.findall(r"\*\*Conflicts in the evidence\*\*(.*?)\*\*",
                                         text, re.S))))
    out["conflicts_reported"] = conflicts
    out["conflicts_expected"] = len(key.get("expected_conflicts", []))
    out["pass"] = (out["allegation_count"] == out["expected_count"]
                   and correct == len(want) and out["summary_consistent"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--report", help="a generated report .md to grade as well")
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text())
    facts = rows("SELECT * FROM triples")
    print(f"case {key['meta']['case_id']}: {len(facts)} extracted fact(s)\n")

    sections = [
        ("entity_resolution", score_entities(key, facts)),
        ("negation_accuracy", score_negation(key, facts)),
        ("date_accuracy", score_dates(key, facts)),
        ("sourcing_discipline", score_sourcing(key, facts)),
        ("fact_recall", score_facts(key, facts)),
    ]
    if args.report:
        sections.append(("report", score_report(key, Path(args.report).read_text())))

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


if __name__ == "__main__":
    sys.exit(main())
