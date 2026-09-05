"""What the corpus contains, whose words it is in, and which allegation owns it.

Report generation has to settle three questions before it writes a word: whether
the corpus is in a state a report can honestly be written from, which assertions
belong to which allegation, and who is speaking in each document. All three are
answered from pipeline state alone - a real investigation has no answer key to
check the corpus against - and all three are inspection rather than prompting,
which is why they sit here instead of adding two hundred lines of SQL to the
module that does the writing.
"""
from __future__ import annotations

import re

from . import paths, state
from .entities import RANKS, normalize
from .extract import allegation_spans
from .log import get_logger

log = get_logger("evidence")

# Every status the runner can leave a document sitting in permanently. Anything
# outside this set - including a status some later stage invents - means the
# document is still moving, which is the same line requeue_unfinished() draws
# when it decides what to pick up again after a restart.
TERMINAL = frozenset({"done", "failed", "incomplete", "text_only"})

# The terminal statuses that mean extraction never ran at all. A document that
# stopped here has transcribed text and nothing else, so none of its content is
# in the graph and none of it can ever be cited. That is a strictly worse
# position than a document that ran and found nothing, which is why it blocks
# rather than warns.
NEVER_ANALYSED = frozenset({"text_only"})

# The runner writes "<n> assertion(s) in the graph" at the moment it marks a
# document done, so the row carries its own witness of what extraction produced.
# Comparing that against what the table holds now is how a corpus that lost
# assertions after processing is told apart from one that never had any.
_RECORDED = re.compile(r"\s*(\d+)\s+assertion", re.I)

_REF_NUMBER = re.compile(r"\d+")

# Form-field labels by which a sworn statement, an interview transcript or a
# declaration names the person whose words follow, most specific first. A closed
# set of English document furniture with no case content in it - the same class
# of structural classifier as ingest._KIND_PATTERNS - and the only way to attach
# a first-person pronoun to a name without asking a model who is speaking. The
# flag marks the labels that name a topic as often as a person; a candidate
# found under one of those has to look like a person in its own right, which is
# the same test extract.interviewee applies to the same labels. The two
# resolvers read the same documents, so a document they disagree about is a
# document whose pronouns two parts of the pipeline attach to different people.
_SPEAKER_FIELDS = [
    (re.compile(r"(?im)^[ \t]*(?:interviewee|person interviewed|deponent|"
                r"affiant|declarant)[ \t]*[:\-][ \t]*(?P<name>[^\n]{2,60})$"),
     False),
    (re.compile(r"(?im)^[ \t]*(?:sworn statement of|statement of|interview of|"
                r"interview with)[ \t]*[:\-]?[ \t]*(?P<name>[^\n]{2,60})$"),
     False),
    (re.compile(r"(?im)^[ \t]*witness[ \t]*[:\-][ \t]*(?P<name>[^\n]{2,60})$"),
     False),
    (re.compile(r"(?im)^[ \t]*(?:name|subject)[ \t]*[:\-][ \t]*"
                r"(?P<name>[^\n]{2,60})$"),
     True),
]

# A middle initial, a rank or an honorific is what separates a "Subject:" line
# holding a rank and a surname from one holding "Complaint against the flight
# chief". Ranks and honorifics live in one list in entities so the two speaker
# resolvers cannot drift apart.
_INITIAL = re.compile(r"\b[A-Za-z]\.")
_WORDS = re.compile(r"[A-Za-z]+")

# A speaker field holds a name and very little else. Anything longer is a
# subject line or a narrative sentence that happens to mention someone, and
# reading a name out of one of those would attach every "I" in the document to
# the wrong person - worse than having no speaker at all.
_MAX_NAME_TOKENS = 6

_HEAD_CHARS = 4000

# A fragment shorter than this is too common to place: a page can hold the same
# eight characters in two different allegations' answers. The same floor
# extract.quote_offset draws, for the same reason.
_MIN_LOCATE_CHARS = 8
# When a passage cannot be found whole, its opening is matched instead and its
# extent is taken from its own length. Matching an opening this long is enough
# to fix where on the page the passage starts.
_PARTIAL_CHARS = 60

# The genres in which an answer can run past the foot of a page and continue
# under no new heading. A log, a memorandum or an attachment is not organised as
# question and answer, so a marker on one of its pages says nothing about the
# next one, and carrying it forward there withholds the rest of the exhibit from
# every allegation except the one the page before it happened to be about.
_CARRY_KINDS = frozenset({"interview", "statement"})

# How many consecutive pages a marker may govern without being restated. One,
# because a wrapped answer continues onto the following page and a marker that
# has gone unrepeated for two whole pages has stopped describing what is on
# them. Running to the end of the document, which is what no cap means, hands
# every appendix to whichever allegation was numbered last.
_MAX_CARRY_PAGES = 1

# Document furniture that opens a new attachment. A page beginning with one of
# these words is a fresh exhibit rather than the continuation of an answer,
# whatever the page before it was about. A closed set of generic English
# structure words with no case content in it, the same class of structural
# classifier as ingest._KIND_PATTERNS.
_ATTACHMENT_HEAD = re.compile(
    r"^(?:tab|exhibit|attachment|enclosure|inclosure|appendix|annex|addendum)"
    r"\b", re.I)

# A heading is short. A line that names an attachment and then keeps going is a
# sentence about one rather than the start of one, and only the heading should
# be allowed to break a continuation.
_MAX_HEADING_CHARS = 80


# -- reading a page a markdown-trained model wrote -------------------------------

# Transcription is done by a text model trained on markdown, so document
# furniture arrives decorated and the decoration lands wherever that model felt
# like putting it. "Interviewee: X", "**Interviewee:** **X**", "- Interviewee:
# X", "### Interviewee: X", "1. Interviewee: X", "> Interviewee: X" and
# "| Interviewee | X |" are one field written seven ways, and an allegation
# heading arrives in the same seven. Widening one anchor at a time is what left
# the last set of forms unread, so decoration is removed once, here, and every
# anchor this module owns or calls reads the undecorated text instead. That
# includes extract.allegation_spans, which is reached only through this function
# and so never has to carry the whole burden in its own pattern. Nothing
# downstream needs an offset into the raw page: the spans, the passage
# placements and the assertion placements are all measured against the
# undecorated text and are only ever compared with each other.

_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+.!|~>-])")

# Emphasis, code and strikethrough marks. Underscore is handled separately
# because it is the one mark that also occurs inside ordinary tokens.
_MD_MARKS = re.compile(r"[*`~]+")
_MD_UNDERSCORE = re.compile(r"(?<![0-9A-Za-z])_+|_+(?![0-9A-Za-z])")

# A blockquote depth, then at most one of a heading, a bullet or an enumerator.
# The bullet and enumerator forms require the space that follows them, so a line
# opening in bold is not mistaken for a bulleted one.
_MD_LEADER = re.compile(r"^(?:[ \t]*>)*[ \t]*(?:#{1,6}[ \t]+|[-*+][ \t]+|"
                        r"\d{1,3}[.)][ \t]+)?")

# The rule row under a table header carries no content. Blanking it rather than
# deleting it keeps the line count intact and leaves the row below it preceded
# by a blank line, which is one of the two things that makes a heading readable
# as a heading rather than as a wrapped sentence.
_MD_RULE = re.compile(
    r"^\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?$")


def _demark_line(line: str) -> str:
    text = _MD_ESCAPE.sub(r"\1", line)
    text = text[_MD_LEADER.match(text).end():]
    text = _MD_UNDERSCORE.sub("", _MD_MARKS.sub("", text))
    stripped = text.strip()
    if _MD_RULE.match(stripped):
        return ""
    # In a pipe table the cell boundary does the work a colon does everywhere
    # else, so it is rewritten as one and every anchor that looks for a label
    # and a value finds them. A leading pipe or two of them, because a single
    # pipe in the middle of a sentence is punctuation rather than a table. A
    # cell's own trailing colon is dropped so that a label written "Name:"
    # inside its cell does not arrive doubled.
    if stripped.startswith("|") or stripped.count("|") >= 2:
        cells = [cell.strip().rstrip(":").strip()
                 for cell in stripped.strip("|").split("|")]
        return ": ".join(cell for cell in cells if cell)
    return text.rstrip()


def _demark(text: str) -> str:
    """The same text with markdown decoration removed, one line for one line.

    One for one: every anchor this feeds is line-anchored, and
    extract._anchor_is_heading reads the line above the one it is testing, so a
    normalisation that merged, dropped or added lines would move a heading into
    the middle of a sentence or invent one at a page break.
    """
    return "\n".join(_demark_line(line) for line in str(text or "").split("\n"))


def _starts_attachment(text: str) -> bool:
    """Whether the page opens a new attachment rather than continuing an answer."""
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        return (len(stripped) <= _MAX_HEADING_CHARS
                and bool(_ATTACHMENT_HEAD.match(stripped)))
    return False


# -- what the corpus is in ------------------------------------------------------


def corpus_state() -> list[dict]:
    """One row per ingested document, with what it has actually contributed.

    dict(row) rather than the sqlite3.Row it comes back as: Row indexes but has
    no .get(), and the miss only shows up at the call site.
    """
    return [dict(r) for r in state.query(
        """SELECT d.doc_id, d.filename, d.status, d.stage, d.doc_kind, d.doc_role,
                  d.progress, d.error,
                  (SELECT COUNT(*) FROM pages p
                    WHERE p.doc_id = d.doc_id AND p.text_path IS NOT NULL)
                    AS pages_with_text,
                  (SELECT COUNT(*) FROM triples t WHERE t.doc_id = d.doc_id)
                    AS assertions
             FROM documents d
            ORDER BY d.uploaded_at""")]


def _recorded_assertions(doc: dict) -> int | None:
    match = _RECORDED.match(doc.get("progress") or "")
    return int(match.group(1)) if match else None


def blocking_faults(docs: list[dict]) -> list[dict]:
    """Documents whose state makes a report unsafe to write at all.

    Three conditions. A document still moving through the queue is evidence
    that has not arrived yet, so findings written now would describe a corpus
    that no longer exists by the time the run ends. A document that was never
    analysed carries content that no finding can cite, however completely it
    was transcribed. A document that finished holding fewer assertions than it
    recorded is pipeline state contradicting itself - the signature of results
    destroyed after the fact - and nothing downstream can tell which findings
    the missing assertions would have changed.

    The recorded count and the present count are compared whenever both are
    known, so a document that kept four of the thirty-seven it recorded is a
    fault exactly as a document that kept none is. The comparison is also what
    keeps the check from crying wolf: a run that recorded nothing and holds
    nothing is not a shortfall, a run whose progress line never carried a count
    cannot be judged either way, and a document that simply extracted nothing
    warns rather than blocks, because a blank cover sheet must not be able to
    deadlock an investigation.
    """
    faults: list[dict] = []
    for doc in docs:
        status = (doc.get("status") or "").strip()
        if status not in TERMINAL:
            faults.append({
                "doc_id": doc["doc_id"], "filename": doc["filename"],
                "reason": f"status {status or 'unknown'} - still queued or "
                          f"processing, so its evidence is not in yet"})
            continue
        if status in NEVER_ANALYSED:
            faults.append({
                "doc_id": doc["doc_id"], "filename": doc["filename"],
                "reason": f"stopped at {status} - its text was transcribed but "
                          f"never analysed, so nothing in it can be cited"})
            continue
        if status != "done":
            continue
        recorded = _recorded_assertions(doc)
        present = int(doc.get("assertions") or 0)
        if recorded is not None and present < recorded:
            faults.append({
                "doc_id": doc["doc_id"], "filename": doc["filename"],
                "reason": f"finished recording {recorded} assertion(s) and "
                          f"holds {present} now - {recorded - present} were "
                          f"lost after processing"})
    return faults


def contributing(contexts: list[dict]) -> set[str]:
    """The documents that actually put something into the assembled context."""
    found: set[str] = set()
    for ctx in contexts:
        for passage in ctx.get("passages") or []:
            if passage.get("doc_id"):
                found.add(passage["doc_id"])
        for fact in ctx.get("facts") or []:
            if fact.get("source_doc"):
                found.add(fact["source_doc"])
    return found


def _arrived_since(docs: list[dict]) -> list[dict]:
    """Documents the corpus gained after the snapshot the report was built on.

    corpus_state() is read once, at the top of a run that then spends minutes in
    the model. A document uploaded during those minutes is in the database and
    in the operator's mind but in none of this report's evidence, and nothing
    else in the pipeline would ever say so. Re-reading the document list at the
    point the integrity warning is assembled is what closes that window.
    """
    known = {d.get("doc_id") for d in docs}
    return [{"doc_id": r["doc_id"], "filename": r["filename"],
             "reason": "ingested after this analysis began, so nothing from it "
                       "could reach these findings"}
            for r in state.query("SELECT doc_id, filename FROM documents")
            if r["doc_id"] not in known]


def silent_documents(docs: list[dict], contributed: set[str]) -> list[dict]:
    """Finished documents that contributed nothing, and why.

    Anything blocking_faults already has is left out - the mid-flight, the never
    analysed, and the document that finished holding fewer assertions than it
    recorded - because naming a document twice in the same warning reads as two
    problems. Which of the two lists it belongs in is not a judgement this
    function makes: it asks blocking_faults, so the two can never disagree about
    where a document is counted.
    """
    blocked = {fault["doc_id"] for fault in blocking_faults(docs)}
    out: list[dict] = []
    for doc in docs:
        status = (doc.get("status") or "").strip()
        if status not in TERMINAL or doc["doc_id"] in blocked:
            continue
        reason = ""
        if status == "failed":
            reason = (f"failed at {doc.get('stage') or 'an unknown stage'}"
                      + (f": {doc['error']}" if doc.get("error") else ""))
        elif status in ("incomplete", "text_only"):
            reason = (doc.get("progress") or doc.get("error")
                      or f"stopped at {status}")
        elif not doc.get("assertions"):
            reason = "finished with no assertions extracted"
        elif doc["doc_id"] not in contributed:
            reason = ("ingested, but no passage or assertion from it reached "
                      "this analysis")
        if reason:
            out.append({"doc_id": doc["doc_id"], "filename": doc["filename"],
                        "reason": reason})
    return out + _arrived_since(docs)


# -- which allegation an assertion belongs to -----------------------------------


def _flat(text) -> str:
    return " ".join(str(text or "").split()).casefold()


def _ref_index(tag) -> int | None:
    """The allegation number inside whatever shape the tag turned out to be.

    Extraction owns that column and may store 3, "3", "3." or "Allegation 3";
    reading the first integer out of it accepts all of those without this module
    having to know which one was chosen.
    """
    if tag is None:
        return None
    match = _REF_NUMBER.search(str(tag))
    return int(match.group(0)) if match else None


def _squash(text: str) -> tuple[str, list[int]]:
    """Lowercased text with whitespace runs collapsed, plus a map from each
    squashed index back to its index in the original.

    A passage is rebuilt from the page by the chunker rather than sliced out of
    it - paragraphs are stripped and rejoined, and a split sentence is carried
    over with a little overlap - so a plain search for a passage in its own page
    misses. Comparing both sides squashed is what makes a passage locatable at
    all, and the index map is what turns the hit back into the character range
    the allegation markers are measured in. Per-character lower() rather than
    casefold(): casefold can return two characters for one, which would break
    the index map that is the whole point of this function.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            low = ch.lower()
            out.append(low if len(low) == 1 else ch)
            idx.append(i)
            prev_space = False
    return "".join(out), idx


# A page that repeated one phrase more times than this is not going to be
# disambiguated by looking at more copies of it, and an unbounded scan of a
# pathological page is a way to hang a report.
_MAX_OCCURRENCES = 32


def _find_all(haystack: str, needle: str) -> list[int]:
    """Every position the needle starts at, overlapping matches included."""
    spots: list[int] = []
    at = 0
    while len(spots) < _MAX_OCCURRENCES:
        pos = haystack.find(needle, at)
        if pos < 0:
            break
        spots.append(pos)
        at = pos + 1
    return spots


def _locate_all(fragment, flat_page: str, idx: list[int]) -> list[tuple[int, int]]:
    """Every character range the fragment occupies in its page, in page order.

    Every, not the first. An interview repeats its short answers - the same
    handful of words once per question - and placing such an answer at the first
    copy files it under whichever allegation was asked first, which is the one
    failure this whole mechanism exists to prevent. Returning all the copies
    lets the caller pick the one its own reading order is up to where it has an
    order to go on, and widen to the union of them where it has none.

    An empty list means "this could not be placed", which every caller has to
    treat as unknown rather than as absent: a fragment whose position is unknown
    is left available to every allegation.
    """
    flat = _squash(_demark(str(fragment or "")))[0].strip()
    if len(flat) < _MIN_LOCATE_CHARS or not idx:
        return []
    spots = _find_all(flat_page, flat)
    if spots:
        return [(idx[pos], idx[min(pos + len(flat) - 1, len(idx) - 1)] + 1)
                for pos in spots]
    if len(flat) <= _PARTIAL_CHARS:
        return []
    # A passage the chunker repaired can differ from the page in a character the
    # squash does not remove, so its opening is matched and its extent taken
    # from its own length. The original is never shorter than the squashed form,
    # so this ends the range at or past its true end - which errs towards
    # covering one marker too many, and an extra allegation number only ever
    # widens what the passage is available to.
    return [(idx[pos], min(idx[pos] + len(str(fragment)), idx[-1] + 1))
            for pos in _find_all(flat_page, flat[:_PARTIAL_CHARS])]


def _page_spans(text: str, inherited: int | None
                ) -> tuple[list[tuple[int, int, int | None]], int | None, list[int]]:
    """The whole page divided into the allegation each stretch of it answers.

    extract.allegation_spans deliberately leaves the text before the first
    marker in no span, which is right when a page is read on its own and wrong
    when it is read in document order: an answer that runs past the foot of one
    page continues at the head of the next, under no new heading, and calling
    that continuation untagged hands it to every allegation at once. The marker
    that governed the end of the previous page therefore governs the head of
    this one until a new marker replaces it. A page with no marker at all is
    covered by whatever it inherited, which is None at the top of a document.

    How far that inheritance is allowed to reach is the caller's decision, not
    this function's: it is handed whatever the caller thinks still governs, and
    _route_from_text stops that at a page cap, at an attachment heading and at
    the genres where an answer cannot continue across a page break at all.
    """
    marks = allegation_spans(text)
    refs = [_ref_index(ref) for _s, _e, ref in marks]
    spans: list[tuple[int, int, int | None]] = []
    if not marks:
        spans.append((0, len(text), inherited))
    else:
        if marks[0][0] > 0:
            spans.append((0, marks[0][0], inherited))
        spans += [(start, end, refs[i])
                  for i, (start, end, _ref) in enumerate(marks)]
    return spans, spans[-1][2], [r for r in refs if r is not None]


def _refs_in(spans: list[tuple[int, int, int | None]], start: int,
             end: int) -> set:
    """Every allegation a character range touches; None where none governs it."""
    hits = {ref for span_start, span_end, ref in spans
            if span_start < end and start < span_end}
    return hits or {None}


def _union_refs(spans: list[tuple[int, int, int | None]],
                spots: list[tuple[int, int]]) -> set:
    """Every allegation any copy of a fragment sits in.

    The union rather than a choice between the copies, because a fragment that
    could be in either place has to stay available in both: withholding it from
    the allegation it belongs to costs more than leaving it in front of one it
    does not, which is the direction this module errs in throughout.
    """
    refs: set = set()
    for spot in spots:
        refs |= _refs_in(spans, *spot)
    return refs


def _note(index: dict, text: str) -> None:
    """Record what routing had to abandon, in the words a report should use.

    Every call site also logs the same sentence at the moment it is produced,
    which is the only way these reach a person today: nothing renders
    index["notes"], so a report of a run whose routing was switched off reads
    exactly like a routed one and only the log says otherwise. Whatever builds
    the corpus integrity warning is where they belong, and until it reads this
    list the notes are an operator-visible gap rather than a disclosure.
    """
    notes = index.setdefault("notes", [])
    if text not in notes:
        notes.append(text)


def _read_page(text_path: str) -> str:
    """The whole page, because marker offsets are measured against all of it.

    A page that cannot be read costs a speaker or an allegation marker, never
    the report: every caller treats the empty string as "nothing is known about
    this page" rather than as "this page said nothing".
    """
    try:
        found = paths.under_root(text_path)
        return found.read_text(encoding="utf-8") if found else ""
    except OSError as exc:
        log.warning("could not read the text of %s: %s", text_path, exc)
        return ""


def _route_from_text(index: dict, triples_by_page: dict, chunks_by_page: dict,
                     kinds: dict) -> tuple[dict, set]:
    """Place every passage and assertion in the allegation span it falls in.

    The markers are read from the page text as well as from the tag extraction
    stored, because that tag was decided one page at a time and so is null for
    every continuation. Reading the pages of a document in order is what lets a
    span cross a page boundary, and reading each passage's own character range
    is what stops a page that turns from one allegation to the next mid-page
    from serving both halves to both.

    Where extraction did record a tag, that tag is one more allegation the
    assertion stays available to, never the only one. Extraction placed the
    quote against the offset of the chunk the model was reading at the time,
    which sounds like information this module does not hold - but a page short
    enough to be a single chunk gives that search the whole page to scan and it
    stops at the first copy, so on exactly the page this mechanism exists to
    handle, one that answers two allegations in the same words, the stored tag
    is the lowest-numbered marker rather than a placement. Reading the page is
    therefore how the untagged remainder gets placed - the continuations, and
    the pages whose markers only this module's undecorated reading can see - and
    also the only check the tag ever gets. The two are unioned, so neither can
    narrow what the other found.
    """
    counts: dict[int, int] = {}
    covered: set = set()
    inherited: int | None = None
    carried = 0
    carry_ok = False
    current_doc = ""
    unreadable = 0
    for row in state.query(
            "SELECT doc_id, page_num, text_path FROM pages "
            "WHERE text_path IS NOT NULL ORDER BY doc_id, page_num"):
        doc, page = row["doc_id"], row["page_num"]
        if doc != current_doc:
            current_doc, inherited, carried = doc, None, 0
            carry_ok = (kinds.get(doc) or "") in _CARRY_KINDS
        raw = _read_page(row["text_path"])
        if not raw:
            # The marker that governed the previous page is carried across an
            # unreadable one: a page we cannot read is not evidence that the
            # questioning moved on.
            unreadable += 1
            continue
        text = _demark(raw)
        if inherited is not None and _starts_attachment(text):
            # A new exhibit answers what it answers on its own terms, so the
            # continuation stops at its heading. The page is then governed by no
            # marker, which leaves it available to every allegation rather than
            # shut inside the one the page before it happened to be about.
            inherited, carried = None, 0
        spans, ends_under, markers = _page_spans(text, inherited)
        for ref in markers:
            counts[ref] = counts.get(ref, 0) + 1
        covered.add((doc, page))
        index["by_page"][(doc, page)] = {ref for _s, _e, ref in spans}
        flat_page, idx = _squash(text)
        cursor = 0
        for chunk in chunks_by_page.get((doc, page)) or []:
            spots = _locate_all(chunk["text"], flat_page, idx)
            ahead = [spot for spot in spots if spot[0] >= cursor]
            if ahead:
                # Passages arrive in the order the chunker cut them, so the copy
                # of a repeated passage that belongs to this one is the first at
                # or after wherever the previous passage was placed. The cursor
                # is advanced one character past the start of the copy just
                # consumed rather than past its end: past its end would let the
                # overlap the chunker deliberately leaves between passages push
                # the next one off the end of the page, while stopping at the
                # start itself would leave that same copy eligible on every
                # later pass, so two passages holding identical text would both
                # be placed at the first occurrence and the second allegation
                # would lose its own copy. One character is far short of the
                # chunker's overlap, so it costs the overlap nothing.
                index["by_chunk"][chunk["chunk_id"]] = _refs_in(spans, *ahead[0])
                cursor = ahead[0][0] + 1
            elif spots:
                # Behind the cursor, so this page's passages are not arriving in
                # page order and there is no reading order left to trust. Every
                # allegation any copy of it sits in keeps the passage.
                index["by_chunk"][chunk["chunk_id"]] = _union_refs(spans, spots)
        for triple in triples_by_page.get((doc, page)) or []:
            stored = _ref_index(triple.get("allegation_ref"))
            spots = _locate_all(triple["quote"], flat_page, idx)
            if spots:
                # Every allegation any copy of the quote sits in, whether the
                # quote was found once or ten times. The stored tag is added to
                # that set rather than replacing it, because the tag is not a
                # better-informed placement: extraction searched the chunk the
                # model was reading, and on a page short enough to be a single
                # chunk that search starts at the head of the page and stops at
                # the first copy, which is the lowest-numbered allegation on the
                # page every time. Letting such a tag override the page-text
                # reading would file both answers to two questions under the
                # question asked first and withhold the second allegation's own
                # answer from it. The union is never narrower than either input,
                # so a disagreement between them widens what the assertion is
                # available to instead of picking a winner.
                refs = _union_refs(spans, spots) | (
                    {stored} if stored is not None else set())
            else:
                # The quote could not be found in its own page text - the
                # chunker repairs a split sentence, and a quote read back from a
                # repaired passage can differ from the page in a character the
                # squash does not remove. That is a failure to place the
                # assertion, not a finding that it belongs anywhere in
                # particular, so it is recorded as unplaced and stays available
                # to every allegation. Writing nothing here instead would drop
                # the assertion through to the page's own markers, which on a
                # fully marked page is a single number, and the assertion would
                # be withheld from every other allegation on the strength of a
                # placement that never happened.
                refs = {None} | ({stored} if stored is not None else set())
            quote = _flat(triple["quote"])
            index["by_quote_pred"].setdefault(
                (doc, page, quote, _flat(triple["predicate"])), set()).update(refs)
            index["by_quote"].setdefault((doc, page, quote), set()).update(refs)
        # What the next page of this document inherits. A page that carried its
        # own markers restarts the count; a page that did not has spent one of
        # the pages a marker is allowed to govern unrepeated, and a document
        # whose genre has no continuations inherits nothing at all.
        if markers:
            inherited, carried = ends_under, 0
        elif inherited is not None:
            carried += 1
            if carried >= _MAX_CARRY_PAGES:
                inherited = None
        if not carry_ok:
            inherited = None
    if unreadable:
        log.warning("%d page(s) could not be read while placing allegation "
                    "markers; assertions on them fall back to the tag "
                    "extraction recorded", unreadable)
    return counts, covered


def _route_from_tags(index: dict, triples_by_page: dict, covered: set,
                     counts: dict) -> None:
    """The stored tag, for pages whose text could not be read back.

    This is the pre-existing behaviour and it is page-granular by construction,
    so it is a fallback rather than the mechanism: it is only consulted where
    the page text that would have done better is gone.
    """
    for (doc, page), rows in triples_by_page.items():
        if (doc, page) in covered:
            continue
        for triple in rows:
            ref = _ref_index(triple["allegation_ref"])
            if ref is not None:
                counts[ref] = counts.get(ref, 0) + 1
            quote = _flat(triple["quote"])
            index["by_quote_pred"].setdefault(
                (doc, page, quote, _flat(triple["predicate"])), set()).add(ref)
            index["by_quote"].setdefault((doc, page, quote), set()).add(ref)
            index["by_page"].setdefault((doc, page), set()).add(ref)


def _refs_present(index: dict) -> set:
    """Every allegation number the index actually places something against.

    Not the same set as the markers counted while reading the pages: an
    assertion can carry a number extraction stored that no marker this module
    could read corroborates. Both have to be checked against the entered
    allegation list, because both can withhold evidence.
    """
    found: set = set()
    for name in ("by_chunk", "by_quote_pred", "by_quote", "by_page"):
        for refs in (index.get(name) or {}).values():
            found |= {ref for ref in refs if ref is not None}
    return found


def _drop_tags(index: dict, unusable: set) -> None:
    """Treat numbers that fit no entered allegation as no number at all.

    Withholding an assertion from every allegation because its number cannot be
    placed is the one outcome routing must never produce: the assertion simply
    disappears from the report, and nothing on the page says it was ever there.
    """
    for name in ("by_chunk", "by_quote_pred", "by_quote", "by_page"):
        table = index.get(name) or {}
        for key, refs in table.items():
            if refs & unusable:
                table[key] = {None if r in unusable else r for r in refs}


def evidence_index(allegation_count: int) -> dict:
    """Allegation tags per assertion and per passage, plus each document's kind.

    Routing is derived from the allegation markers in the page text, so it works
    on a database that predates the allegation_ref migration; where a stored tag
    exists it widens what its assertion is available to and never narrows it,
    and it decides the placement alone only on pages whose text is no longer
    readable. graph.py takes the same line on event_date_basis. A corpus whose documents carry no markers
    yields no tags, routing is left off, and every assertion stays available to
    every allegation - the behaviour that existed before any of this.
    """
    docs = {r["doc_id"]: dict(r) for r in state.query(
        "SELECT doc_id, doc_kind, doc_role FROM documents")}
    index = {
        "enabled": False,
        "kind": {k: (v.get("doc_kind") or "unknown") for k, v in docs.items()},
        "role": {k: (v.get("doc_role") or "") for k, v in docs.items()},
        "by_chunk": {}, "by_quote_pred": {}, "by_quote": {}, "by_page": {},
        "zero_based": False, "notes": [],
    }

    columns = {r["name"] for r in state.query("PRAGMA table_info(triples)")}
    ref_column = ("allegation_ref" if "allegation_ref" in columns
                  else "NULL AS allegation_ref")
    triples_by_page: dict = {}
    for row in state.query(f"SELECT doc_id, page_num, predicate, quote, "
                           f"{ref_column} FROM triples"):
        triples_by_page.setdefault(
            (row["doc_id"], row["page_num"]), []).append(dict(row))
    # In page order, because _route_from_text places a repeated passage at the
    # copy that follows the one before it and has no order of its own to fall
    # back on: chunk_id is a hash, so the ord column is the only thing that
    # says which passage the chunker cut first.
    chunks_by_page: dict = {}
    for row in state.query("SELECT chunk_id, doc_id, page_num, text FROM chunks "
                           "ORDER BY doc_id, page_num, ord"):
        chunks_by_page.setdefault(
            (row["doc_id"], row["page_num"]), []).append(dict(row))

    counts, covered = _route_from_text(index, triples_by_page, chunks_by_page,
                                       index["kind"])
    _route_from_tags(index, triples_by_page, covered, counts)

    seen = set(counts)
    if not seen:
        log.info("no allegation markers were found in the corpus; every "
                 "assertion stays available to every allegation")
        return index
    index["enabled"] = True

    # Whether the numbering counts allegations from zero or from one is a
    # property of the documents, so it is read off them rather than assumed -
    # but only when the whole scheme corroborates it. A single stray zero, which
    # one mis-transcribed heading is enough to produce, would otherwise shift
    # every allegation onto the previous one's evidence.
    one_based = set(range(1, allegation_count + 1))
    zero_based = set(range(0, allegation_count))
    if seen <= one_based:
        usable = one_based
    elif (0 in seen and counts.get(0, 0) > 1 and seen <= zero_based
            and max(seen) == allegation_count - 1):
        index["zero_based"] = True
        usable = zero_based
        log.info("allegation markers read as zero-based (%s)", sorted(seen))
    else:
        usable = one_based

    # The number a document writes beside an allegation is the number that
    # document uses. Nothing guarantees it is the position that allegation holds
    # in the list the operator entered - a referral can send on two of four
    # allegations, in the order the appointing memo lists them - and routing on
    # an unverified correspondence delivers each allegation another one's
    # evidence with no warning at all. So the numbers are checked against the
    # list, and where they do not fit, routing is abandoned rather than guessed.
    stray = seen - usable
    unplaceable = {tag for tag in stray if counts[tag] > 1}
    if unplaceable:
        index["enabled"] = False
        _note(index, f"Allegation routing was switched off. The documents number "
                     f"their allegations {sorted(seen)}, which does not fit the "
                     f"{allegation_count} allegation(s) entered, so no assertion "
                     f"could be placed against one; every allegation below was "
                     f"analysed against the whole record.")
        log.error("allegation markers %s do not fit the %d allegation(s) "
                  "entered; routing is off and every allegation is analysed "
                  "against the whole record", sorted(seen), allegation_count)
        return index
    # Everything that survived to here and still fits no entered allegation is
    # freed, whether it was read off a page or carried in on an assertion's own
    # stored tag. Counting only the markers would leave a stored tag outside the
    # entered range matching no allegation at all, and _available() reads that
    # as "withheld from all of them": the assertion would then vanish from the
    # report with nothing on the page to say it was ever there, which is the one
    # outcome routing must never produce.
    unusable = _refs_present(index) - usable
    if unusable:
        _drop_tags(index, unusable)
        _note(index, f"Allegation number(s) {sorted(unusable)} match no entered "
                     f"allegation; what they mark was left available to every "
                     f"allegation rather than withheld from all of them.")
        log.warning("allegation number(s) %s match no entered allegation; "
                    "treating them as untagged", sorted(unusable))
    log.info("allegation markers %s routed onto %d entered allegation(s)",
             sorted(seen), allegation_count)
    return index


def _available(refs, target: int) -> bool:
    """An unplaced assertion is unrestricted; a tag we cannot place is kept.

    Ambiguity keeps evidence. Withholding a fact from the allegation it belongs
    to costs more than leaving one in front of an allegation it does not. None
    in the set means the stretch of page it came from is governed by no marker
    and by no marker carried into it, which is genuinely unrestricted rather
    than merely untagged.
    """
    if not refs:
        return True
    return None in refs or target in refs


def _keep_fact(fact: dict, target: int, index_map: dict) -> bool:
    doc, page = fact.get("source_doc"), fact.get("source_page")
    if not doc or page is None:
        return True
    quote = _flat(fact.get("quote"))
    refs = index_map["by_quote_pred"].get(
        (doc, page, quote, _flat(fact.get("predicate"))))
    if refs is None:
        refs = index_map["by_quote"].get((doc, page, quote))
    if refs is None:
        refs = index_map["by_page"].get((doc, page))
    return _available(refs, target)


def _keep_passage(passage: dict, target: int, index_map: dict) -> bool:
    """Route a passage on the stretch of page it occupies, not on its page.

    Questioning turns from one allegation to the next in the middle of a page,
    so a page-granular decision serves one allegation's answer to the other. A
    passage that could not be placed in its page text falls back to the page,
    and only where every stretch of that page is governed by some marker: a page
    that is partly unrestricted tells us nothing about a passage we could not
    find on it.
    """
    refs = index_map["by_chunk"].get(passage.get("chunk_id"))
    if refs is None:
        page_refs = index_map["by_page"].get(
            (passage.get("doc_id"), passage.get("page_num")))
        if not page_refs or None in page_refs:
            return True
        refs = page_refs
    return _available(refs, target)


def route(passages: list[dict], facts: list[dict], allegation_index: int,
          index_map: dict) -> tuple[list[dict], list[dict]]:
    """Withhold evidence marked as testimony about a different allegation."""
    if not index_map.get("enabled"):
        return passages, facts
    target = (allegation_index - 1 if index_map.get("zero_based")
              else allegation_index)
    kept_passages = [p for p in passages if _keep_passage(p, target, index_map)]
    kept_facts = [f for f in facts if _keep_fact(f, target, index_map)]
    if (passages or facts) and not kept_passages and not kept_facts:
        # A numbering that does not line up with the allegation list must never
        # be able to produce a finding written from no evidence at all.
        _note(index_map,
              f"Allegation {allegation_index} was analysed against the whole "
              f"record: every passage and assertion retrieved for it is marked "
              f"as testimony about some other allegation, which cannot be "
              f"right, so the marks were ignored for it.")
        log.error("allegation %d: routing would have removed every passage and "
                  "fact, so routing is skipped for it", allegation_index)
        return passages, facts
    return kept_passages, kept_facts


def enrich(facts: list[dict], index_map: dict) -> list[dict]:
    """Put the document kind and the speaker's role back onto each graph fact.

    graph.entity_detail does not return r.source_kind or r.source_role even
    though the loader writes both, so every fact has been arriving with them
    empty and the record-first ordering in the report has had nothing to sort
    on. The values are restored from the same documents rows the loader read
    them from, so this reproduces what the edge already holds.
    """
    kinds = index_map.get("kind") or {}
    roles = index_map.get("role") or {}
    for fact in facts:
        doc = fact.get("source_doc")
        if not fact.get("source_kind"):
            fact["source_kind"] = kinds.get(doc) or "unknown"
        if not fact.get("source_role"):
            fact["source_role"] = roles.get(doc) or ""
    return facts


# -- who is speaking ------------------------------------------------------------


def _read_head(text_path: str) -> str:
    """A missing or unreadable transcript costs a speaker, never the report."""
    return _read_page(text_path)[:_HEAD_CHARS]


def _person_shaped(raw: str) -> bool:
    """Whether a value under a topical label reads as a person in its own right.

    "Subject:" and "Name:" head a topic at least as often as a person, so a value
    found under one has to carry something only a name carries - a rank, an
    honorific, or a middle initial.
    """
    if _INITIAL.search(raw):
        return True
    return any(word in RANKS for word in _WORDS.findall(raw.casefold()))


def _name_tokens(norm: str) -> list[str]:
    """The parts of a normalised name that identify anyone.

    A lone letter is a middle initial. Forms write one ("Surname, Given I.") and
    entity resolution does not keep it, so comparing it either way would only
    ever make two spellings of one person look like two people.
    """
    return [token for token in norm.split() if len(token) > 1]


def _surname(norm: str, canonical: str) -> str:
    """Which part of an extracted name a field has to write in order to place it.

    Not simply the last one. Forms write names in both orders, entity resolution
    keeps whichever spelling reached it first, and its tie-break can leave the
    surname-first spelling as the survivor, so treating the last part as the
    surname places nobody at all for half the corpus: the document gets no
    speaker, its "I" is never attached to anyone, and the prompts fall back to
    unresolved pronouns - the false-conflict failure this module was built to
    remove. The comma is the only thing that says which order a name is in and
    normalise() strips it, so the canonical spelling is read for it rather than
    the normalised form. Widening to "either end of the name" instead would
    accept a bare given name, which places nobody in a different and worse way.
    """
    parts = _name_tokens(norm)
    if not parts:
        return ""
    head, comma, _rest = str(canonical or "").partition(",")
    if comma:
        surname_parts = _name_tokens(normalize(head))
        if surname_parts:
            return surname_parts[-1]
    return parts[-1]


def _match_person(candidate: str, people: list[tuple[str, str]]) -> str:
    """The one extracted person whose name is the whole of this field, or "".

    Whole, not contained: a field reading "Complaint against <name>" contains a
    person and is not that person's statement, and reading it as one attaches
    every "I" in a complainant's memo to the person complained about. The field
    may write less of the name than the entity does, and may write it in either
    order - a surname alone, or "Last, First", is a normal way to fill one in -
    but it may not write anything else, and it must write the surname, since a
    bare first name places nobody. Two different people answering to the same
    field means the field places nobody either.
    """
    tokens = set(_name_tokens(candidate))
    if not tokens:
        return ""
    hits = []
    for norm, canonical in people:
        parts = _name_tokens(norm)
        surname = _surname(norm, canonical)
        if parts and surname and tokens <= set(parts) and surname in tokens:
            hits.append((parts, canonical))
    if not hits:
        return ""
    # Entity resolution may still be holding "Surname" and "Given Surname"
    # apart. Those
    # are one person written at two lengths, and the longer is the more specific
    # answer; names that are not nested that way are different people.
    best = max(hits, key=lambda hit: len(hit[0]))
    if all(set(parts) <= set(best[0]) for parts, _c in hits):
        return best[1]
    log.info("a speaker field names %r, which matches %d extracted people; "
             "leaving the document without a speaker",
             candidate, len({c for _p, c in hits}))
    return ""


def _speaker_in(text: str, people: list[tuple[str, str]]) -> str:
    """The speaker named by the most specific form field the head of the page has.

    The text is undecorated first, so that all seven ways a markdown-trained
    transcriber can write one field reach the same anchors rather than each
    needing its own tolerance bolted onto four regexes.
    """
    text = _demark(text)
    for pattern, topical in _SPEAKER_FIELDS:
        for match in pattern.finditer(text):
            raw = match.group("name")
            if topical and not _person_shaped(raw):
                continue
            candidate = normalize(raw)
            if not candidate or len(candidate.split()) > _MAX_NAME_TOKENS:
                continue
            found = _match_person(candidate, people)
            if found:
                return found
    return ""


def speakers() -> dict[str, str]:
    """doc_id -> the canonical name of the person whose words the document is.

    A document only enters this map when its form field resolves to a person the
    pipeline already extracted, which is what stops a subject line about a topic
    from being read as a speaker. Documents that do not resolve are simply
    absent, and the prompts are told not to resolve pronouns in a document with
    no speaker rather than to guess at one - an unresolved "I" produces false
    conflicts between accounts that actually agree.
    """
    people = [(normalize(r["canonical_name"]), r["canonical_name"])
              for r in state.query(
                  "SELECT canonical_name FROM entities "
                  "WHERE merged_into IS NULL AND entity_type='PERSON'")]
    people = [(n, c) for n, c in people if len(n) >= 3]
    if not people:
        return {}

    found: dict[str, str] = {}
    seen: set[str] = set()
    for row in state.query(
            "SELECT doc_id, page_num, text_path FROM pages "
            "WHERE text_path IS NOT NULL ORDER BY doc_id, page_num"):
        if row["doc_id"] in seen:
            continue
        seen.add(row["doc_id"])
        text = _read_head(row["text_path"])
        if not text:
            continue
        name = _speaker_in(text, people)
        if name:
            found[row["doc_id"]] = name
    return found
