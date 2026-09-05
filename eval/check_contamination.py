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
facts allegation allegations investigation investigator record records""".split())


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

    # Person names, and every distinctive token inside them: a surname or a
    # callsign on its own is exactly what leaked.
    for person in key.get("entities", {}).get("persons", key.get("entities", [])) or []:
        name = person.get("name") if isinstance(person, dict) else str(person)
        if not name:
            continue
        # A "name" built only from ordinary words ("The Subject") is not
        # evidence of copying, and treating it as a needle would flag any
        # prompt that uses the word subject.
        distinctive = [tok for tok in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", name)
                       if tok.lower() not in COMMON]
        if distinctive:
            found.add(name)
            found |= set(distinctive)
        for alias in (person.get("aliases") or []) if isinstance(person, dict) else []:
            found.add(str(alias))
    return {n for n in found if len(n) >= 4}


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
        print("no keys found; nothing to check against")
        return 0

    needles: dict[str, set[str]] = {}
    for kp in keys:
        try:
            needles[kp.name] = needles_from_key(json.loads(kp.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  could not read {kp.name}: {exc}")

    hits = []
    for rel in PROMPT_MODULES:
        path = ROOT / rel
        if not path.exists():
            continue
        for lineno, text in model_facing_strings(path):
            low = text.lower()
            for key_name, terms in needles.items():
                for term in terms:
                    # Word boundaries, so "4 May" does not match inside
                    # "14 May" and report a leak that is not there.
                    m = re.search(rf"(?<![\w$]){re.escape(term.lower())}(?!\w)",
                                  low)
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



CONTROLS = [
    ("a surname lifted from a key is caught",
     {"entities": {"persons": [{"name": "MSgt Owen T. Brandt"}]}},
     'When the text says a person is called something, as in "MSgt Brandt, who '
     'everyone calls Ox", emit an assertion.', True),
    ("a callsign in an alias list is caught",
     {"entities": {"persons": [{"name": "Owen Brandt", "aliases": ["Oxlike"]}]}},
     "the witness known as Oxlike said nothing", True),
    ("a day-and-month example is caught",
     {"gold_facts": [{"id": "F1", "text": "the move happened on 18 April 2026"}]},
     "A statement dated 8 June describing something on 18 April takes the "
     "earlier date.", True),
    ("a money figure is caught",
     {"gold_facts": [{"id": "F1", "text": "charges totaling $412.88 posted"}]},
     "for example a restaurant charge of $412.88", True),
    ("a document id is caught",
     {"documents": [{"doc_id": "C2_09", "file_match": "C2_09_Interview_Osei"}]},
     "cite the source document, for example C2_09, in every assertion", True),
    ("a bare year is NOT reported",
     {"gold_facts": [{"id": "F1", "text": "during 2026 the flight moved"}]},
     "dates are written YYYY-MM-DD, for example 2026-03-14", False),
    ("a clock time is NOT reported",
     {"gold_facts": [{"id": "F1", "text": "the meeting began at 0900"}]},
     'convert 24-hour times: "0900" becomes T09:00', False),
    ("an ordinary English word in a name is NOT reported",
     {"entities": {"persons": [{"name": "The Subject"}]}},
     "the subject of your assertion must be a named person", False),
]


def self_test() -> int:
    print("negative controls - the guard must fire on a real leak "
          "and stay quiet otherwise\n")
    passed = 0
    for name, key, prompt, should_fire in CONTROLS:
        terms = needles_from_key(key)
        low = prompt.lower()
        fired = any(
            re.search(rf"(?<![\w$]){re.escape(t.lower())}(?!\w)", low)
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
