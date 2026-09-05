"""Extract subject-predicate-object assertions from each page's text.

Every assertion has to carry a quote that actually appears on the page it
claims to come from.  Assertions whose quote cannot be located are discarded:
that check is what keeps an invented fact with a plausible-looking citation out
of the graph, and it costs one string search.
"""
from __future__ import annotations

import hashlib
import json
import re

from rapidfuzz import fuzz

from . import paths, state
from .config import env_int, env_str
from .entities import entity_id, normalize
from .log import get_logger, utcnow
from .model_client import Ollama, default_options, thinking_enabled
from .prompts_extract import ENTITY_TYPES, build as build_prompt

log = get_logger("extract")

_WS = re.compile(r"\s+")
QUOTE_FUZZ_THRESHOLD = 90
MAX_QUOTE_CHARS = 400
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?$")
# Models write military times back in several shapes: "2026-05-14 0925",
# "2026-05-14T0925", "2026-05-14 09:25". Times are evidence in their own right -
# a vehicle leaving at 1012 and returning at 1540 is the substance of a finding -
# so these are repaired rather than discarded.
_LOOSE_DT = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ]?(\d{2}):?(\d{2})(?::?(\d{2}))?$")


def normalize_when(value: str) -> str:
    """Return an ISO date or date-time, or "" if it cannot be trusted."""
    text = str(value or "").strip().replace("Z", "")
    if not text or text.lower() in {"null", "none", "unknown", "n/a", "n/a."}:
        return ""
    if DATE_RE.match(text):
        return text.replace(" ", "T", 1)
    loose = _LOOSE_DT.match(text)
    if loose:
        day, hh, mm = loose.group(1), int(loose.group(2)), int(loose.group(3))
        if hh <= 23 and mm <= 59:
            return f"{day}T{hh:02d}:{mm:02d}"
        return day
    # A bare date hiding inside something longer is still usable.
    bare = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return bare.group(0) if bare else ""
# A page of dense text is chunked so the model sees the whole page even on a
# modest context window.
CHUNK_CHARS = 6000


def triple_id(doc_id: str, page_num: int, parts: tuple[str, ...]) -> str:
    payload = "|".join([doc_id, str(page_num), *parts]).casefold()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _flat(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


_PRONOUNS = {"i", "me", "you", "he", "she", "they", "we", "the interviewee",
             "interviewee", "unknown", "the subject", "the witness", "io",
             "the member", "the investigator"}

# A role is a position, not a person. Extracting "Equipment Test Craftsman" as
# a PERSON creates a node that can silently collect another person's actions -
# the same class of harm as an invented name, reached by a different route.
# Plurals and collectives - "Airmen", "the crew", "the shop" - name groups,
# not people. Same harm as a job title: a node that accumulates the actions of
# whichever individual the model could not pin down.
_COLLECTIVE = re.compile(
    r"^(the\s+)?(airmen|airman|personnel|members|member|troops|crew|shop|team|"
    r"staff|squadron|flight|section|unit|others|everyone|someone|somebody|"
    r"people|witnesses|subordinates|coworkers|colleagues|leadership|"
    r"supervisors|management)$", re.I)

_ROLE_WORDS = re.compile(
    r"\b(administrator|technician|craftsman|journeyman|apprentice|chief|"
    r"supervisor|monitor|manager|officer|commander|custodian|operator|"
    r"specialist|analyst|inspector|superintendent|director|noncommissioned|"
    r"nco|ncoic|oic|section|flight|squadron|shop|clerk|assistant)\b", re.I)
_HONORIFIC = re.compile(
    r"^(a1c|amn|sra|ssgt|tsgt|msgt|smsgt|cmsgt|2lt|1lt|capt|maj|lt|ltcol|col|"
    r"gen|mr|mrs|ms|miss|dr|sir|madam|civ)\b", re.I)


def looks_like_role(name: str) -> bool:
    """True when a PERSON name is really a job title.

    A real name may legitimately contain a role word after a rank ("TSgt Chief"
    is a surname); the test is whether anything name-like survives once rank
    and role vocabulary are removed.
    """
    text = str(name or "").strip()
    if _COLLECTIVE.match(text):
        return True
    if _HONORIFIC.match(text):
        return False
    if not _ROLE_WORDS.search(text):
        return False
    remainder = _ROLE_WORDS.sub(" ", text)
    remainder = re.sub(r"\b(the|a|an|of|and|civilian|test|equipment|network)\b",
                       " ", remainder, flags=re.I)
    return not re.search(r"[A-Za-z]{3}", remainder)


def name_grounded(name: str, page_text: str, header: str) -> bool:
    """A hard grounding constraint: no entity without a verbatim source span.

    A graded run produced a person who exists in no document - actions from
    three different people were hung on an invented name. The prompt now
    forbids that, but a prompt is a request; this is the enforcement. The name
    (rank stripped, whitespace collapsed) must appear in the page text or the
    document header, or the assertion is dropped with the reason recorded.
    """
    flat = _WS.sub(" ", str(name or "")).strip().casefold()
    if not flat or flat in _PRONOUNS:
        return False
    haystack = _WS.sub(" ", f"{page_text}\n{header}").casefold()
    if flat in haystack:
        return True
    # "SSgt Smith" is grounded by a page that says just "Smith": strip rank
    # words and try the bare name, but require something longer than initials.
    from .entities import RANKS
    tokens = [t for t in flat.replace(".", " ").split() if t not in RANKS]
    bare = " ".join(tokens)
    return bool(bare) and len(bare) >= 3 and bare in haystack


def quote_supported(quote: str, page_text: str) -> bool:
    q, t = _flat(quote), _flat(page_text)
    if not q or len(q) < 8:
        return False
    if q in t:
        return True
    # Whitespace carries no meaning for this check, and OCR invents it: a table
    # cell read as "OUT BOUND" is the same evidence as "OUTBOUND", and quoting
    # it the natural way should not count as an unsupported citation.
    squashed_q = _WS.sub("", q)
    if len(squashed_q) >= 8 and squashed_q in _WS.sub("", t):
        return True
    # Beyond that, tolerate only small drift such as line-wrap hyphenation.
    return fuzz.partial_ratio(q, t) >= QUOTE_FUZZ_THRESHOLD


def salvage_objects(text: str) -> list[dict]:
    """Pull whole JSON objects out of a truncated reply.

    A reply cut off at the token cap is not parseable as a whole, but the
    assertions it did finish are perfectly good.  Discarding them because the
    last one was clipped would throw away a page's work.
    """
    out: list[dict] = []
    starts: list[int] = []          # one entry per open brace, at any depth
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            start = starts.pop()
            # The assertions sit inside a {"triples": [...]} wrapper, so the
            # objects worth recovering close at depth 1, not depth 0.
            try:
                item = json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "subject_name" in item:
                out.append(item)
    return out


def parse_response(raw: str) -> list[dict]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        salvaged = salvage_objects(text)
        if salvaged:
            log.warning("reply was not valid JSON (likely cut off at the token "
                        "cap); recovered %d complete assertion(s)", len(salvaged))
            return salvaged
        return []
    if isinstance(data, list):
        return [t for t in data if isinstance(t, dict)]
    if isinstance(data, dict):
        for key in ("triples", "results", "assertions", "data"):
            if isinstance(data.get(key), list):
                return [t for t in data[key] if isinstance(t, dict)]
        if "subject_name" in data:
            return [data]
    return []


def _clean_name(value, entity_type: str) -> str:
    """Drop a type prefix the model repeated inside the name.

    The type travels in its own field, so "CLAIM: he was there" would otherwise
    become a node literally called "CLAIM: he was there".
    """
    name = str(value).strip()
    for prefix in (f"{entity_type}:", f"{entity_type} :"):
        if name.upper().startswith(prefix.upper()):
            name = name[len(prefix):].strip()
            break
    return name[:200]


def validate(item: dict, page_text: str, header: str = "") -> tuple[dict | None, str]:
    for field in ("subject_type", "subject_name", "predicate",
                  "object_type", "object_name", "quote"):
        if not str(item.get(field) or "").strip():
            return None, f"missing {field}"

    # People and organisations must be traceable to ink on a page. CLAIM and
    # EVENT names are the model's own summarising label, so they are exempt.
    for role, type_field in (("subject_name", "subject_type"),
                             ("object_name", "object_type")):
        etype = str(item.get(type_field) or "").strip().upper()
        if etype in ("PERSON", "ORG") and not name_grounded(
                str(item[role]), page_text, header):
            return None, f"{role} {item[role]!r} does not appear in the document"
        if etype == "PERSON" and looks_like_role(str(item[role])):
            return None, f"{role} {item[role]!r} is a job title, not a person"

    subject_type = str(item["subject_type"]).strip().upper()
    object_type = str(item["object_type"]).strip().upper()
    if subject_type not in ENTITY_TYPES or object_type not in ENTITY_TYPES:
        return None, "unknown entity type"

    quote = str(item["quote"]).strip()[:MAX_QUOTE_CHARS]
    if not quote_supported(quote, page_text):
        return None, "quote does not appear on the page"

    event_date = normalize_when(item.get("event_date"))
    basis = str(item.get("event_date_basis") or "").strip().lower()
    if basis not in ("stated", "month", "approx", "inferred"):
        basis = "stated" if event_date else ""
    if not event_date:
        basis = ""

    # A name that grounded but still carries first person is a leak, not a fact.
    for role in ("subject_name", "object_name"):
        low = str(item[role]).lower()
        if low.startswith(("my ", "our ")) or " my " in f" {low} ":
            return None, f"{role} contains unresolved first person: {item[role]!r}"

    return {
        "subject_type": subject_type,
        "subject_name": _clean_name(item["subject_name"], subject_type),
        "predicate": str(item["predicate"]).strip().lower()[:100],
        "object_type": object_type,
        "object_name": _clean_name(item["object_name"], object_type),
        "event_date": event_date or None,
        "event_date_basis": basis or None,
        "quote": quote,
    }, ""


def chunks(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, current = [], []
    size = 0
    for para in text.split("\n\n"):
        if size + len(para) > CHUNK_CHARS and current:
            out.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
    if current:
        out.append("\n\n".join(current))
    return out


def register_entity(conn, entity_type: str, name: str) -> None:
    conn.execute(
        """INSERT INTO entities (entity_id, entity_type, canonical_name,
                                 first_seen, mention_count)
           VALUES (?,?,?,?,1)
           ON CONFLICT(entity_id) DO UPDATE SET
             mention_count = entities.mention_count + 1""",
        (entity_id(entity_type, name), entity_type, name.strip(), utcnow()))


def _migrate() -> None:
    for stmt in ("ALTER TABLE triples ADD COLUMN event_date_basis TEXT",
                 "ALTER TABLE documents ADD COLUMN doc_kind TEXT",
                 "ALTER TABLE documents ADD COLUMN doc_role TEXT"):
        try:
            with state.tx() as conn:
                conn.execute(stmt)
        except Exception:
            pass    # column already there


def run(doc_id: str, on_progress) -> tuple[int, int]:
    _migrate()
    rows = state.query(
        """SELECT doc_id, page_num, text_path FROM pages
           WHERE doc_id=? AND text_path IS NOT NULL ORDER BY page_num""", (doc_id,))
    if not rows:
        return 0, 0

    client = Ollama()
    model = client.require_model(env_str("TEXT_MODEL", ""), "TEXT_MODEL")
    options = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                              "EXTRACT_NUM_PREDICT", 1200)

    # Page 1's opening usually carries the metadata header that names the
    # interviewee - the anchor for resolving "I" and "you" on every page.
    header = ""
    first = paths.under_root(rows[0]["text_path"]) if rows else None
    if first is not None:
        header = first.read_text(encoding="utf-8")[:500]

    kept = dropped = 0
    for idx, row in enumerate(rows, 1):
        on_progress(f"extracting from page {idx}/{len(rows)} with {model}")
        text_file = paths.under_root(row["text_path"])
        if text_file is None:
            continue
        page_text = text_file.read_text(encoding="utf-8").strip()
        if not page_text or page_text == "[no text]":
            continue

        for chunk in chunks(page_text):
            system, user, version = build_prompt(doc_id, row["page_num"], chunk,
                                                 header=header)
            try:
                data = client.generate(model, user, system=system, options=options,
                                       format_json=True, think=thinking_enabled())
            except Exception as exc:
                log.error("%s p%s: %s", doc_id, row["page_num"], exc)
                continue

            for item in parse_response(data.get("response") or ""):
                clean, reason = validate(item, chunk, header=header)
                if clean is None:
                    dropped += 1
                    log.info("%s p%d: dropped (%s)", doc_id, row["page_num"], reason)
                    continue
                tid = triple_id(doc_id, row["page_num"],
                                (clean["subject_type"], clean["subject_name"],
                                 clean["predicate"], clean["object_type"],
                                 clean["object_name"]))
                with state.tx() as conn:
                    conn.execute(
                        """INSERT INTO triples (triple_id, doc_id, page_num,
                             subject_type, subject_name, predicate, object_type,
                             object_name, event_date, event_date_basis, quote,
                             model, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(triple_id) DO NOTHING""",
                        (tid, doc_id, row["page_num"], clean["subject_type"],
                         clean["subject_name"], clean["predicate"], clean["object_type"],
                         clean["object_name"], clean["event_date"],
                         clean["event_date_basis"], clean["quote"],
                         f"{model} ({version})", utcnow()))
                    register_entity(conn, clean["subject_type"], clean["subject_name"])
                    register_entity(conn, clean["object_type"], clean["object_name"])
                kept += 1

    log.info("%s: %d assertion(s) kept, %d dropped for unsupported quotes",
             doc_id, kept, dropped)
    return kept, dropped


def rebuild_entities() -> int:
    """Recompute the entities table from the triples that still exist.

    Deleting a document used to remove its triples but leave its entity rows,
    so a person from a deleted file went on being offered to chat and reports
    as a known name - and a graded run showed the model then hanging other
    people's actions on that phantom. Entities are derived data; after any
    deletion they are rebuilt from scratch and re-merged.
    """
    with state.tx() as conn:
        conn.execute("DELETE FROM entities")
    rows = state.query(
        "SELECT subject_type, subject_name, object_type, object_name FROM triples")
    with state.tx() as conn:
        for r in rows:
            register_entity(conn, r["subject_type"], r["subject_name"])
            register_entity(conn, r["object_type"], r["object_name"])
    auto_merge()
    return len(rows)


def auto_merge(on_progress=lambda _m: None) -> int:
    """Fold obvious name variants together: Smith / SSgt Smith / J. Smith.

    Runs automatically.  Only high-confidence forms merge - an initialism or a
    surname against a full name, or an exact match after rank and punctuation
    are stripped.  Anything less certain is left as separate entities, because
    wrongly joining two people is worse than showing two nodes.
    """
    from itertools import combinations

    threshold = float(env_int("MERGE_THRESHOLD", 88))
    rows = state.query(
        "SELECT entity_id, entity_type, canonical_name, mention_count FROM entities "
        "WHERE merged_into IS NULL ORDER BY entity_type, mention_count DESC")

    by_type: dict[str, list] = {}
    for row in rows:
        by_type.setdefault(row["entity_type"], []).append(row)

    merged = 0
    for entity_type, group in by_type.items():
        if entity_type not in {"PERSON", "ORG", "LOCATION"} or len(group) < 2:
            continue
        norms = {r["entity_id"]: normalize(r["canonical_name"]) for r in group}
        for a, b in combinations(group, 2):
            score = _score(norms[a["entity_id"]], norms[b["entity_id"]])
            if score < threshold:
                continue
            # The better-attested name survives.
            keep, lose = (a, b) if a["mention_count"] >= b["mention_count"] else (b, a)
            if state.query_one("SELECT merged_into FROM entities WHERE entity_id=?",
                               (lose["entity_id"],))["merged_into"]:
                continue
            with state.tx() as conn:
                conn.execute("UPDATE entities SET merged_into=? WHERE entity_id=?",
                             (keep["entity_id"], lose["entity_id"]))
            merged += 1
            log.info("merged %s into %s (%.0f)", lose["canonical_name"],
                     keep["canonical_name"], score)
    if merged:
        on_progress(f"merged {merged} duplicate name(s)")
    return merged


def _score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    a_tokens, b_tokens = a.split(), b.split()
    short, long = sorted((a_tokens, b_tokens), key=lambda t: len(" ".join(t)))
    flat = "".join(short)
    if flat.isalpha() and len(flat) <= 4 and flat == "".join(t[0] for t in long):
        return 92.0     # "j s" against "john smith"
    if (len(a_tokens) == 1 or len(b_tokens) == 1) and a_tokens[-1] == b_tokens[-1]:
        return 90.0     # "smith" against "john smith"
    return float(fuzz.token_sort_ratio(a, b))
