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


def threshold(key: dict, category: str, default: int) -> int:
    """Read the pass bar out of the key. Hardcoding it here would mean the
    harness quietly grades every case against whatever the last case needed.
    """
    raw = str(key.get("scoring", {}).get(category, {}).get("pass_threshold", ""))
    m = re.search(r"(\d+)\s*/\s*\d+", raw)
    return int(m.group(1)) if m else default


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
    NEG = re.compile(r"\b(never|not|n't|no|zero|none|failed|denied|refused|"
                     r"without|absent|missing|unable)\b", re.I)

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
    # overlap alone cannot tell "first casino cash advance" from the second,
    # third and fourth - one extracted advance would satisfy all four gold
    # entries and report recall that was never earned. A timeline is a claim
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
            if sum(w in blob for w in words) < need:
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
        # Keys identify documents differently ("doc 09", "C2_09"); read
        # whichever form this key uses rather than assuming one.
        m = re.search(r"primary source is (?:doc\s*)?([A-Za-z0-9_]+)", trap, re.I)
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


def score_conflicts(key: dict, text: str) -> dict:
    """Which expected conflicts the report actually surfaced.

    A conflict counts as detected when the report's conflict sections name
    enough of both positions to show it understood the disagreement - not
    merely that the words appear somewhere in the document.
    """
    sections = re.findall(r"\*\*Conflicts in the evidence\*\*(.*?)(?:\*\*Gaps|\Z)",
                          text, re.S)
    blob = norm(" ".join(sections))
    detected, missed = [], []
    for c in key.get("expected_conflicts", []):
        sides = list(c.get("positions", {}).values())
        hits = 0
        for side in sides:
            words = [w for w in norm(side).split() if len(w) > 4][:5]
            if words and sum(w in blob for w in words) >= max(1, len(words) // 3):
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
    # Checking co-presence passed a case where "Ox" stood as its own separate
    # entity beside Brandt - the exact split the trap exists to catch.
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


def _money_forms(value: str) -> list[str]:
    """Every way a document might legitimately write one amount.

    A verbatim transcript says "412 dollars and 88 cents" where the key writes
    $412.88. Both are the same figure faithfully recorded, and a check that
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
    traps = " ".join(key.get("trap_inventory", {}).get("exculpatory_routing", []))
    ids = re.findall(r"\bF\d{2}\b", traps)
    gold = {g["id"]: g["text"] for g in key["gold_facts"]}
    body = norm(text)
    present = []
    for fid in ids:
        words = [w for w in norm(gold.get(fid, "")).split() if len(w) > 4][:6]
        if words and sum(w in body for w in words) >= max(2, len(words) // 3):
            present.append(fid)
    return {"exculpatory": f"{len(present)}/{len(ids)}", "found": present,
            "missing": [i for i in ids if i not in present],
            "pass": bool(ids) and len(present) >= max(1, len(ids) - 1)}


def score_credibility(key: dict, text: str) -> dict:
    """Motive/credibility surfaced, and hearsay chains denied probative weight."""
    body = norm(text)
    credibility = any(w in body for w in
                      ("credibility", "motive", "bias", "grievance", "lor",
                       "letter of reprimand", "reliability"))
    hearsay = any(w in body for w in
                  ("hearsay", "secondhand", "second hand", "unnamed",
                   "could not name", "not firsthand", "not first hand"))
    return {"credibility_discussed": credibility, "hearsay_flagged": hearsay,
            "pass": credibility and hearsay}


def score_report(key: dict, text: str) -> dict:
    """Disposition accuracy, allegation count, and summary/findings agreement."""
    out: dict = {}
    # The disposition is a single line. With re.S a trailing (.+) swallows the
    # rest of the document, so the first block matches once and the other two
    # are never seen - the check reported one allegation where three exist.
    blocks = re.findall(
        r"#### Allegation (\d+):(.*?)\*\*Disposition:\*\*[ \t]*([^\n]+)",
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
    if "--self-test" in sys.argv:
        return self_test()
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
        ("merge_split", score_merges(key, facts)),
        ("numeric_fidelity", score_numeric(key, facts)),
    ]
    if args.report:
        report_text = Path(args.report).read_text()
        sections.append(("conflict_recall", score_conflicts(key, report_text)))
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

SCORER_CONTROLS = [
    ("timeline rejects the right event on the wrong date",
     lambda: not score_dates(
         {"timeline_gold": [{"date": "2026-06-12", "precision": "day",
                             "event": "First casino cash advance"}],
          "scoring": {"date_accuracy": {"pass_threshold": ">=1/1"}}},
         [{"quote": "cash advance at the casino", "predicate": "made",
           "object_name": "advance", "subject_name": "Brandt",
           "event_date": "2026-07-03", "event_date_basis": "stated"}])["pass"]),

    ("timeline refuses to let one event satisfy two ordinals",
     lambda: score_dates(
         {"timeline_gold": [
             {"date": "2026-06-12", "precision": "day", "event": "First casino cash advance"},
             {"date": "2026-07-03", "precision": "day", "event": "Second casino cash advance"}],
          "scoring": {"date_accuracy": {"pass_threshold": ">=2/2"}}},
         [{"quote": "cash advance at the casino", "predicate": "made",
           "object_name": "advance", "subject_name": "Brandt",
           "event_date": "2026-06-12", "event_date_basis": "stated"}]
     )["timeline_events"] == "1/2"),

    ("numeric accepts an amount written in words",
     lambda: score_numeric(
         {"gold_facts": [{"id": "F10", "text": "restaurant charges of $412.88"}],
          "scoring": {"numeric_fidelity": {"metric": "F10 numeric fields exact"}}},
         [{"quote": "totaling 412 dollars and 88 cents", "object_name": ""}])["pass"]),

    ("numeric still fails on a changed digit",
     lambda: not score_numeric(
         {"gold_facts": [{"id": "F10", "text": "restaurant charges of $412.88"}],
          "scoring": {"numeric_fidelity": {"metric": "F10 numeric fields exact"}}},
         [{"quote": "totaling 412 dollars and 89 cents", "object_name": ""}])["pass"]),

    ("sourcing fails when only the secondary document is cited",
     lambda: not score_sourcing(
         {"documents": [{"doc_id": "C2_09", "file_match": "C2_09_Interview_Osei"},
                        {"doc_id": "C2_11", "file_match": "C2_11_Interview_Brandt"}],
          "gold_facts": [{"id": "F10",
                          "text": "nine transactions totaling 1,847 dollars"}],
          "trap_inventory": {"sourcing": ["F10 primary source is C2_09; "
                                          "restatements in C2_11 are secondary"]}},
         [{"quote": "nine transactions totaling 1,847 dollars", "object_name": "",
           "doc_id": "C2_11_Interview_Brandt_Subject"}])["pass"]),

    ("sourcing passes when the document of record is cited",
     lambda: score_sourcing(
         {"documents": [{"doc_id": "C2_09", "file_match": "C2_09_Interview_Osei"}],
          "gold_facts": [{"id": "F10",
                          "text": "nine transactions totaling 1,847 dollars"}],
          "trap_inventory": {"sourcing": ["F10 primary source is C2_09"]}},
         [{"quote": "nine transactions totaling 1,847 dollars", "object_name": "",
           "doc_id": "C2_09_Interview_Osei_GTC"}])["pass"]),

    ("threshold is read from the key, not from this file",
     lambda: threshold({"scoring": {"date_accuracy":
                                    {"pass_threshold": ">=19/22"}}},
                       "date_accuracy", default=0) == 19),
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
