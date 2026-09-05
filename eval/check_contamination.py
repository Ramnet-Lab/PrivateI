"""Fail if an answer key's vocabulary appears in a prompt sent to the model.

Three separate times, a worked example in a prompt turned out to be lifted from
the case being graded. The extraction prompt carried "MSgt Brandt, who everyone
calls Ox" while the key's graded fact F01 read "known by callsign Ox"; it
carried "since August 2024" against a key fact reading "since Aug 2024"; and it
carried a boundary-date example taken from the other case entirely. Each one
handed the model the answer it was about to be scored on, and each survived
because a person read the prompt and did not recognise the example.

So this does not rely on reading. It takes the distinctive strings out of every
key and looks for them in every string the model is given. Run it in CI, before
grading, and after any prompt edit.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ROOT / "eval" / "keys"

# Modules whose string literals reach the model. A comment is not sent, so
# comments are deliberately out of scope; docstrings are skipped for the same
# reason.
PROMPT_MODULES = [
    "app/pipeline/prompts_extract.py",
    "app/pipeline/prompts_transcribe.py",
    "app/pipeline/report.py",
    "app/pipeline/chat.py",
    "app/pipeline/transcribe.py",
]

MONTHS = ("january february march april may june july august september "
          "october november december").split()
_MONTH_RE = "|".join(MONTHS + [m[:3] for m in MONTHS])

# A needle has to be distinctive enough that its presence is evidence of
# copying rather than coincidence. "May" and "the" are useless; "11 May",
# "$412.88" and a surname are not.
COMMON = set("""the and for with that this from have been were was are not but
all any one two three four five six seven eight nine ten first second third
other some such only into over under after before since until date time page
statement report interview witness subject officer person people evidence fact
facts allegation allegations investigation investigator record records
msgt ssgt tsgt smsgt cmsgt sra amn a1c capt maj col gen sgt lieutenant
colonel major captain sergeant airman master senior chief technical staff
mister missus doctor""".split())


def needles_from_key(key: dict) -> set[str]:
    found: set[str] = set()
    blob = json.dumps(key)

    # Explicit dates in any of the forms a document or a key might write.
    found |= set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", blob))
    found |= {m.group(0) for m in
              re.finditer(rf"\b\d{{1,2}}\s+(?:{_MONTH_RE})\w*\b", blob, re.I)}
    found |= {m.group(0) for m in
              re.finditer(rf"\b(?:{_MONTH_RE})\w*\s+\d{{4}}\b", blob, re.I)}
    # Money and other multi-digit figures.
    found |= set(re.findall(r"\$[\d,]+(?:\.\d{2})?", blob))
    # A number is only distinctive when it is grouped or fractional. A bare
    # four-digit run is a year or a 24-hour clock time, and a prompt has to be
    # able to teach the format of both without being accused of copying one.
    found |= set(re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b", blob))
    found |= set(re.findall(r"\b\d+\.\d{2}\b", blob))
    # Document identifiers.
    for d in key.get("documents", []):
        for field in ("doc_id", "file_match"):
            if d.get(field):
                found.add(str(d[field]))

    # Person names. Both real keys nest these under entities.persons and store
    # the name as "canonical", not "name" - an earlier version read only "name",
    # got None for every person, and reported a clean prompt having checked no
    # name at all.
    #
    # Only groups that actually hold names are mined. "forbidden_entities_
    # examples" contains strings like "any name not listed above", and reading
    # it as a roster produced needles for the words name, listed and above,
    # which then matched almost every prompt in the repo.
    ents = key.get("entities")
    people: list = []
    if isinstance(ents, dict):
        for group_name, group in ents.items():
            if not isinstance(group, list) or "forbidden" in group_name:
                continue
            people += group
    elif isinstance(ents, list):
        people = ents

    short: set[str] = set()
    for person in people:
        if isinstance(person, str):
            person = {"canonical": person}
        if not isinstance(person, dict):
            continue
        forms = [person.get("canonical"), person.get("name")]
        forms += list(person.get("aliases") or [])
        for form in forms:
            if not isinstance(form, str) or not form.strip():
                continue
            # A peripheral mention is written "Maj Elena V. Cross (appointing
            # authority)"; the gloss in parentheses is description, not name.
            form = re.sub(r"\(.*?\)", " ", form).strip(" ,")
            if not form:
                continue
            # A name token is capitalised where the key writes it. That single
            # test removes the descriptive words - answers, authority, speaker -
            # without needing to enumerate them.
            distinctive = [t for t in re.findall(r"\b[A-Z][A-Za-z'-]{2,}", form)
                           if t.lower() not in COMMON]
            if distinctive:
                found.add(form)
                found |= set(distinctive)
            # A callsign can be two letters. "Ox" is the exact string that
            # leaked into the extraction prompt, and a four-character floor
            # threw it away. Short forms are matched with word boundaries so
            # they cannot fire inside a longer word.
            if 2 <= len(form) <= 3 and form[:1].isupper() and form.lower() not in COMMON:
                short.add(form)

    # Regulations, orders and other cited authorities are distinctive strings a
    # prompt has no reason to contain.
    blob_text = " ".join(str(v) for v in (key.get("allegations") or [])) + " " + blob
    found |= set(re.findall(r"\b(?:AFI|OI|DAFI|AFMAN|DoDI|DoD)\s*[\d][\d.\-]*",
                            blob_text))
    found |= set(re.findall(r"\bArticle\s+\d+\b", blob_text))

    return {n for n in found if len(n) >= 4} | short


def model_facing_strings(path: Path) -> list[tuple[int, str]]:
    """Every string literal the module hands to the model, minus docstrings."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings and len(node.value) >= 8):
            out.append((node.lineno, node.value))
    return out


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    keys = sorted(KEYS.glob("*.json"))
    if not keys:
        print("  NO KEYS FOUND. This check cannot pass without something to "
              "check against; a missing corpus is not a clean result.")
        return 1

    needles: dict[str, set[str]] = {}
    unreadable: list[str] = []
    for kp in keys:
        try:
            needles[kp.name] = needles_from_key(json.loads(kp.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  UNREADABLE KEY {kp.name}: {exc}")
            unreadable.append(kp.name)

    if unreadable:
        print(f"  {len(unreadable)} key(s) could not be read, so their "
              f"vocabulary went unchecked. Failing rather than reporting a "
              f"clean result from a partial corpus.")
        return 1

    hits = []
    for rel in PROMPT_MODULES:
        path = ROOT / rel
        if not path.exists():
            continue
        for lineno, text in model_facing_strings(path):
            for key_name, terms in needles.items():
                for term in terms:
                    # Word boundaries, so "4 May" does not match inside
                    # "14 May" and report a leak that is not there.
                    m = re.search(rf"(?<![\w$]){re.escape(term)}(?!\w)", text)
                    if m:
                        start = max(0, m.start() - 45)
                        hits.append((rel, lineno, key_name, term,
                                     text[start:m.end() + 45].replace("\n", " ")))

    print(f"checked {len(PROMPT_MODULES)} prompt module(s) against "
          f"{len(keys)} key(s)\n")
    if not hits:
        print("  clean: no key vocabulary appears in any prompt")
        return 0
    for rel, lineno, key_name, term, excerpt in hits:
        print(f"  CONTAMINATED {rel}:{lineno}")
        print(f"    matches {key_name} term {term!r}")
        print(f"    in: {excerpt}")
    print(f"\n  {len(hits)} contamination(s). A prompt must not contain the "
          f"answer it will be graded on.")
    return 1



# Controls use the REAL key shape (entities.persons[].canonical + aliases).
# An earlier version tested a {"persons": [{"name": ...}]} shape that no key on
# disk uses, so all eight controls passed against a code path that never ran on
# real data - the guard reported a clean prompt having checked no name at all.
def _key(persons=None, facts=None, docs=None):
    out = {}
    if persons:
        out["entities"] = {"persons": persons}
    if facts:
        out["gold_facts"] = facts
    if docs:
        out["documents"] = docs
    return out


_REAL_SHAPE = [{"id": "brandt", "canonical": "MSgt Owen T. Brandt",
                "aliases": ["Brandt", "MSgt Brandt", "Ox", "the NCOIC"]}]

CONTROLS = [
    ("the exact leak that motivated this guard is caught",
     _key(persons=_REAL_SHAPE),
     'STATED NICKNAMES ARE FACTS. When the text says a person is called '
     'something else - "MSgt Brandt, who everyone calls Ox" - emit an '
     'assertion.', True),
    ("a two-letter callsign is caught on its own",
     _key(persons=_REAL_SHAPE),
     "for example a witness everyone calls Ox", True),
    ("a surname is caught on its own",
     _key(persons=_REAL_SHAPE),
     "resolve the pronoun to Brandt where the header names him", True),
    ("a canonical name is caught",
     _key(persons=_REAL_SHAPE),
     "the subject of the assertion is MSgt Owen T. Brandt", True),
    ("a day-and-month example is caught",
     _key(facts=[{"id": "F1", "text": "the move happened on 18 April 2026"}]),
     "A statement dated 8 June describing something on 18 April takes the "
     "earlier date.", True),
    ("a money figure is caught",
     _key(facts=[{"id": "F1", "text": "charges totaling $412.88 posted"}]),
     "for example a restaurant charge of $412.88", True),
    ("a document id is caught",
     _key(docs=[{"doc_id": "C2_09", "file_match": "C2_09_Interview_Osei"}]),
     "cite the source document, for example C2_09, in every assertion", True),
    ("a cited regulation is caught",
     {"allegations": [{"text_key": "conduct contrary to Article 93"}]},
     "an allegation may cite Article 93 of the code", True),
    ("a bare year is NOT reported",
     _key(facts=[{"id": "F1", "text": "during 2026 the flight moved"}]),
     "dates are written YYYY-MM-DD, for example 2026-03-14", False),
    ("a clock time is NOT reported",
     _key(facts=[{"id": "F1", "text": "the meeting began at 0900"}]),
     'convert 24-hour times: "0900" becomes T09:00', False),
    ("an ordinary English word in a name is NOT reported",
     _key(persons=[{"canonical": "The Subject"}]),
     "the subject of your assertion must be a named person", False),
    ("a short callsign does NOT fire inside a longer word",
     _key(persons=_REAL_SHAPE),
     "the assertion must be unambiguous and fully grounded in the page", False),
]


def self_test() -> int:
    print("negative controls - the guard must fire on a real leak "
          "and stay quiet otherwise\n")
    passed = 0
    for name, key, prompt, should_fire in CONTROLS:
        terms = needles_from_key(key)
        fired = any(re.search(rf"(?<![\w$]){re.escape(t)}(?!\w)", prompt)
                    for t in terms)
        ok = fired == should_fire
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         expected fire={should_fire}, got {fired}; "
                  f"needles={sorted(terms)}")
        passed += ok
    print(f"\n  {passed}/{len(CONTROLS)} controls behaved correctly")
    return 0 if passed == len(CONTROLS) else 1


if __name__ == "__main__":
    sys.exit(main())
