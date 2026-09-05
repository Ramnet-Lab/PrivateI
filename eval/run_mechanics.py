#!/usr/bin/env python3
"""Run the capability fixtures against a live instance and score the mechanics.

Each fixture is a tiny document that isolates one mechanic - negation,
coreference, grounding, date precision, hearsay - and a set of assertions over
the facts extracted from it. The assertions are properties, not answers: they
would hold for any case, which is the point. A suite that only knows the case
it was built from stops being a test the moment the case changes.

    python3 eval/run_mechanics.py                 # all fixtures
    python3 eval/run_mechanics.py negation        # one mechanic
    python3 eval/run_mechanics.py --keep          # leave fixtures uploaded

Exit code is the number of failed fixtures, so CI can gate on it.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
APP = "http://127.0.0.1:8080"
PREFIX = "evalfx_"


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def api(path: str) -> dict:
    out = sh("curl", "-s", "--max-time", "30", f"{APP}{path}")
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


def upload(fixture: dict) -> str:
    """Write the fixture as a .txt the app will read as a text document."""
    tmp = ROOT / "eval" / "fixtures" / f"{PREFIX}{fixture['id']}.docx"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    _write_docx(tmp, fixture["text"])
    out = sh("curl", "-s", "-X", "POST", f"{APP}/upload", "-F", f"files=@{tmp}")
    try:
        data = json.loads(out)
        return data["accepted"][0]["doc_id"] if data.get("accepted") else ""
    except (json.JSONDecodeError, KeyError, IndexError):
        return ""


def _write_docx(path: Path, text: str) -> None:
    """Minimal .docx so the app's Word reader handles it - no dependency on the
    fixture author having python-docx installed outside the container."""
    import zipfile
    paras = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{_xml(line)}</w:t></w:r></w:p>'
        for line in text.splitlines())
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           f'<w:body>{paras}</w:body></w:document>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
          'relationships+xml"/><Override PartName="/word/document.xml" ContentType='
          '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '</Types>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc)


def _xml(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def wait_idle(timeout: int = 900) -> bool:
    waited = 0
    while waited < timeout:
        counts = api("/api/documents").get("counts", {})
        if counts.get("processing", 1) == 0:
            return True
        time.sleep(5)
        waited += 5
    return False


def triples_for(doc_id: str) -> list[dict]:
    db = ROOT / "data" / "state.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("SELECT * FROM triples WHERE doc_id=?", (doc_id,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# --- assertions -----------------------------------------------------------

def check(rule: dict, rows: list[dict]) -> tuple[bool, str]:
    names = [r["subject_name"] for r in rows] + [r["object_name"] for r in rows]
    people = {r["subject_name"] for r in rows if r["subject_type"] == "PERSON"} | \
             {r["object_name"] for r in rows if r["object_type"] == "PERSON"}

    if "forbid_entity" in rule:
        pat = re.compile(rule["forbid_entity"], re.I)
        hit = [n for n in names if pat.search(n) or pat.match(n.lower())]
        return (not hit), f"forbidden entity present: {hit}" if hit else ""
    if "require_entity" in rule:
        pat = re.compile(rule["require_entity"], re.I)
        return (any(pat.search(n) for n in names),
                f"no entity matching /{rule['require_entity']}/ in {sorted(set(names))}")
    if "entity_roster_max" in rule:
        n = len(people)
        return (n <= rule["entity_roster_max"],
                f"{n} people extracted, max {rule['entity_roster_max']}: {sorted(people)}")
    if "entity_roster_min" in rule:
        n = len(people)
        return (n >= rule["entity_roster_min"],
                f"only {n} people extracted, need {rule['entity_roster_min']}: {sorted(people)}")
    if "forbid_predicate" in rule:
        pat = re.compile(rule["forbid_predicate"], re.I)
        hit = [r["predicate"] for r in rows if pat.search(r["predicate"])]
        return (not hit), f"forbidden predicate present: {hit}" if hit else ""
    if "require_predicate" in rule:
        pat = re.compile(rule["require_predicate"], re.I)
        return (any(pat.search(r["predicate"]) for r in rows),
                f"no predicate matching /{rule['require_predicate']}/ in "
                f"{[r['predicate'] for r in rows]}")
    if "forbid_date" in rule:
        where = re.compile(rule.get("where_predicate", ".*"), re.I)
        hit = [f"{r['predicate']} @ {r['event_date']}" for r in rows
               if (r["event_date"] or "").startswith(rule["forbid_date"])
               and where.search(r["predicate"])]
        return (not hit), f"forbidden date used: {hit}" if hit else ""
    if "forbid_exact_stated_date" in rule:
        where = re.compile(rule.get("where_predicate", ".*"), re.I)
        hit = [f"{r['predicate']} @ {r['event_date']} ({r.get('event_date_basis')})"
               for r in rows
               if where.search(r["predicate"]) and (r["event_date"] or "")
               and len(r["event_date"]) >= 10
               and (r.get("event_date_basis") or "stated") == "stated"]
        return (not hit), f"day-precise date claimed as stated: {hit}" if hit else ""
    if "require_quote_substr" in rule:
        needle = rule["require_quote_substr"].lower()
        return (any(needle in (r["quote"] or "").lower() for r in rows),
                f"no quote containing {needle!r}")
    return True, ""


# --- negative controls ------------------------------------------------------
# A suite that has never failed is decoration. These feed each rule the exact
# defect it exists to catch and require it to fire, so a green run means the
# pipeline is clean rather than the checks being asleep.
NEGATIVE_CONTROLS = [
    ("forbid_entity fires on a pronoun entity",
     {"forbid_entity": r"^(i|me|my|the interviewee|unknown)$"},
     [{"subject_name": "I", "subject_type": "PERSON", "object_name": "the fault",
       "object_type": "EVENT", "predicate": "reported", "quote": "I reported it",
       "event_date": None, "event_date_basis": None}]),
    ("require_entity fires when the interviewee is absent",
     {"require_entity": r"Okonkwo"},
     [{"subject_name": "Someone Else", "subject_type": "PERSON", "object_name": "x",
       "object_type": "EVENT", "predicate": "did", "quote": "q",
       "event_date": None, "event_date_basis": None}]),
    ("forbid_predicate fires on a flipped negation",
     {"forbid_predicate": r"^replied to$"},
     [{"subject_name": "TSgt Kerr", "subject_type": "PERSON", "object_name": "the email",
       "object_type": "DOCUMENT", "predicate": "replied to",
       "quote": "never replied", "event_date": None, "event_date_basis": None}]),
    ("entity_roster_max fires on an invented third person",
     {"entity_roster_max": 2},
     [{"subject_name": f"Person {i}", "subject_type": "PERSON", "object_name": "x",
       "object_type": "EVENT", "predicate": "did", "quote": "q",
       "event_date": None, "event_date_basis": None} for i in range(3)]),
    ("forbid_date fires on a boundary date used as an event date",
     {"forbid_date": "2026-05-11", "where_predicate": r"zero|no |failed|missing"},
     [{"subject_name": "log", "subject_type": "DOCUMENT", "object_name": "checks",
       "object_type": "EVENT", "predicate": "shows zero", "quote": "zero after 11 May",
       "event_date": "2026-05-11", "event_date_basis": "stated"}]),
    ("forbid_exact_stated_date fires on a sharpened approximation",
     {"forbid_exact_stated_date": True, "where_predicate": r"been"},
     [{"subject_name": "TSgt Nunez", "subject_type": "PERSON", "object_name": "NCOIC",
       "object_type": "CLAIM", "predicate": "has been", "quote": "since March 2026",
       "event_date": "2026-03-01", "event_date_basis": "stated"}]),
    ("require_quote_substr fires when attribution is dropped",
     {"require_quote_substr": "told me"},
     [{"subject_name": "SSgt Lane", "subject_type": "PERSON", "object_name": "A1C Reed",
       "object_type": "PERSON", "predicate": "shouted at", "quote": "Lane shouted at him",
       "event_date": None, "event_date_basis": None}]),
]


def self_test() -> int:
    print("negative controls - every rule must fire on the defect it targets\n")
    bad = 0
    for name, rule, rows_in in NEGATIVE_CONTROLS:
        ok, _why = check(rule, rows_in)
        fired = not ok
        print(f"  [{'PASS' if fired else 'BROKEN'}] {name}")
        if not fired:
            bad += 1
            print("           rule did NOT fire - this check cannot detect its defect")
    print(f"\n  {len(NEGATIVE_CONTROLS) - bad}/{len(NEGATIVE_CONTROLS)} controls fired")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Run capability fixtures")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the checks fire on known-bad data, then exit")
    ap.add_argument("mechanic", nargs="?", help="run only this mechanic")
    ap.add_argument("--keep", action="store_true", help="leave fixtures uploaded")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    fixtures = yaml.safe_load((ROOT / "eval" / "mechanics.yaml").read_text())
    if args.mechanic:
        fixtures = [f for f in fixtures if f["mechanic"] == args.mechanic]
    if not fixtures:
        print("no fixtures selected")
        return 0

    if not api("/api/documents"):
        print("the app is not answering on 127.0.0.1:8080 - start it first")
        return 1

    print(f"uploading {len(fixtures)} fixture(s)")
    ids: dict[str, str] = {}
    for f in fixtures:
        doc_id = upload(f)
        if not doc_id:
            print(f"  {f['id']}: upload failed")
        ids[f["id"]] = doc_id

    print("waiting for extraction")
    if not wait_idle():
        print("timed out waiting for processing")

    failures = 0
    by_mechanic: dict[str, list[bool]] = {}
    print()
    for f in fixtures:
        doc_id = ids.get(f["id"], "")
        rows = triples_for(doc_id) if doc_id else []
        problems = []
        for rule in f["assert"]:
            ok, why = check(rule, rows)
            if not ok:
                problems.append(why)
        passed = not problems and bool(rows)
        if not rows:
            problems.append("no facts extracted at all")
        by_mechanic.setdefault(f["mechanic"], []).append(passed)
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {f['id']:<24} {f['mechanic']:<14} ({len(rows)} facts)")
        if not passed:
            failures += 1
            for p in problems:
                print(f"         {p}")
            print(f"         why it matters: {f['why']}")

    print()
    for mech, results in sorted(by_mechanic.items()):
        print(f"  {mech:<16} {sum(results)}/{len(results)}")
    print(f"\n  {len(fixtures) - failures}/{len(fixtures)} fixtures passed")

    if not args.keep:
        for doc_id in ids.values():
            if doc_id:
                sh("curl", "-s", "-X", "POST", f"{APP}/documents/{doc_id}/delete")
        for p in (ROOT / "eval" / "fixtures").glob(f"{PREFIX}*"):
            p.unlink()
    return failures


if __name__ == "__main__":
    sys.exit(main())
