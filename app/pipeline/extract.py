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

from . import llm_settings, paths, state
from .config import env_int
from .entities import RANKS, entity_id, normalize
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

# The calendar and the grammar of English dates, which is what decides how sharp
# a recorded date is allowed to be. Nothing here names a case, a person, or a
# year: a month name is a month name in any investigation.
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
# Spelled out in full rather than as three-letter prefixes: a prefix followed
# by "any letters" turns "Marceau" into March and "Decision" into December, and
# a false month is enough to demote a real day to a month.
_MONTH_ALT = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
              r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|"
              r"nov(?:ember)?|dec(?:ember)?")
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})\b\.?,?\s+(\d{{4}})\b", re.I)
_MONTH_NAME = re.compile(rf"\b(?:{_MONTH_ALT})\b", re.I)
_DAY_BEFORE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:{_MONTH_ALT})\b", re.I)
_DAY_AFTER = re.compile(
    rf"\b(?:{_MONTH_ALT})\b\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)
_DAY_ORDINAL = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", re.I)
# English writes a day in words as often as in digits, and a quote that says
# "the fourteenth of March" states its day as firmly as "14 March" does.
# The ordinals are a closed class of the language, so listing them names no
# case. "the" is required in front so that "a third party" and "the second
# time" - counters rather than dates - do not offer a day.
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30}
_DAY_WORD = re.compile(
    r"\bthe\s+(?:(twenty|thirty)[\s-]"
    r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)"
    r"|(" + "|".join(_ORDINAL_WORDS) + r"))\b", re.I)
_DAY_NUMERIC = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
# A clock time, in either civil or military form. A civil time is
# self-identifying because of its colon; a bare four-digit number is not, and a
# document corpus is full of four-digit numbers that are order numbers, room
# numbers and product codes. The military form therefore has to be introduced
# by a preposition that takes a time complement, or followed by the unit that
# names it as a time. That is a grammatical requirement rather than a guess at
# which documents are duty logs, so it holds in a corpus of any genre.
_CLOCK = re.compile(
    r"\b\d{1,2}:[0-5]\d\b"
    r"|\b(?:at|around|about|approximately|by|until|since|from|before|after)\s+"
    r"(?<![\d.$])(?:[01]\d|2[0-3])[0-5]\d\b(?!\.\d)"
    r"|(?<![\d.$])\b(?:[01]\d|2[0-3])[0-5]\d\s*(?:hrs?|hours|[LZ])\b"
    r"|\b(?:noon|midnight|[ap]\.?m\.?)\b", re.I)
# Hedges are a closed grammatical class of English approximators, marking the
# speaker's own uncertainty. A hedge only hedges the DATE when it modifies a
# time expression: "about the payment on 31 January" is not an approximate
# date, and neither is "six late arrivals", so the word alone is not enough. A
# bare small integer is not a time expression either - "about 12 requests" and
# "roughly 20 minutes later" both leave a stated day exactly as stated - so the
# number has to carry an ordinal suffix or a month beside it to count.
_HEDGE = re.compile(
    r"\b(?:around|about|approximately|roughly|sometime|somewhere|mid|early|late)"
    r"[\s-]+(?:(?:in|of|the|on)\s+)?(?:(?:mid|early|late)[\s-]+)?"
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)\b|\d{{1,2}}[\s-]+(?:{_MONTH_ALT})\b|"
    rf"(?:{_MONTH_ALT})\b|(?:19|20)\d{{2}}\b|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|"
    r"week|month|year|morning|afternoon|evening|night)"
    r"|\bor so\b|\bthereabouts\b", re.I)

# The model that writes these replies is markdown-trained, and so is the OCR of
# a page that was itself rendered from markdown, so every anchor in this module
# has to survive decoration that the writer of a regex did not picture: bold
# around a label, a heading hash or a bullet in front of it, a numbered list, a
# blockquote marker, a whole line rendered as a pipe-table row, and emphasis on
# both sides of a colon at once. Rather than teach each anchor its own escape
# sequences one bug at a time, decoration is described once here and every
# anchor is built out of these three pieces: _MD_LEAD is whatever may stand
# between the start of a line and its first real word, _MD_WRAP is emphasis
# pressed directly against a word on either side, and _undecorate() strips both
# off a value that has already been cut out of a line. A decoration that is not
# handled yet therefore has one place to be added rather than six.
_MD_MARKS = r"*_~`"
_MD_WRAP = rf"[{_MD_MARKS}]*"
_MD_LEAD = (rf"[ \t>|]*(?:(?:#{{1,6}}|[-+\u2022]|\d{{1,3}}[.)])[ \t]*)*"
            rf"(?:[{_MD_MARKS}]+[ \t]*)*")
# A line that is furniture rather than a sentence: a table separator, a
# horizontal rule, a row of underscores on a form. It ends where it ends, so
# nothing wraps out of it into the line below.
_RULE_LINE = re.compile(rf"[\s|:=.{_MD_MARKS}\u2013\u2014+-]*")


def _undecorate(value: str) -> str:
    """A line or a field value with its markdown decoration removed.

    Used wherever a value has been cut out of a decorated line and then has to
    be parsed or compared as plain text: "**A. Nakamura**" and "| A. Nakamura |"
    are the same name as "A. Nakamura", and a name-shape test that has never
    seen an asterisk rejects two of the three.
    """
    text = _WS.sub(" ", str(value or "")).strip()
    text = re.sub(rf"^{_MD_LEAD}", "", text)
    return re.sub(rf"[\s|{_MD_MARKS}]+$", "", text).strip()


def _line_is_open(line: str) -> bool:
    """True when a line runs on into the one below it.

    OCR wraps a sentence at the page width, so the start of a line is not the
    start of a sentence and a marker sitting there may be the middle of
    somebody's answer. Only running prose wraps: a blank line, a rule, a table
    row and a line closing with terminal punctuation all end where they end.
    Decoration is stripped before the last character is read, because
    "**...and he moved on.**" ends the sentence it ends whatever is wrapped
    around it.
    """
    raw = str(line or "").strip()
    if raw.startswith("|") or raw.endswith("|"):
        return False
    text = _undecorate(raw)
    if not text or _RULE_LINE.fullmatch(text):
        return False
    return text[-1] not in ".!?:;\u2013\u2014"


# An investigation questions one allegation at a time, and the line introducing
# the next one is where the previous subject ends. Matched at line start only,
# with an optional speaker tag ("IO:", "Q:", "CAPT WREN:") in front: a witness
# who says "under this allegation" mid-sentence is still talking, not turning
# the page. "charge" and "count" are deliberately not anchor words - a ledger
# line ("Charge 3 posted 6 February") would false-anchor on them. A line start
# is only a candidate: _anchor_is_heading decides which of these markers
# actually opens a section, because OCR wraps sentences and a wrapped line
# begins in the middle of somebody's answer.
# Every part of the marker is allowed its own decoration, because the model
# puts it in different places on different lines: around the whole heading,
# around the keyword alone, around the number alone, or on both sides of the
# colon between them. A pipe closes the marker the way a full stop does, so a
# table row that opens with the marker is read as one.
_ALLEGATION_ANCHOR = re.compile(
    rf"(?mi)^{_MD_LEAD}(?:[A-Z][A-Za-z. ]{{0,14}}{_MD_WRAP}:{_MD_WRAP}[ \t]*)?"
    rf"{_MD_WRAP}(?:allegation|specification){_MD_WRAP}[ \t]*"
    rf"(?::{_MD_WRAP}[ \t]*)?{_MD_WRAP}(?:no\.?|#)?[ \t]*{_MD_WRAP}"
    rf"(\d{{1,2}}|[ivx]{{1,4}})\b{_MD_WRAP}[ \t]*[.:)|\u2013\u2014-]?")
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7,
          "viii": 8, "ix": 9, "x": 10}


def normalize_when(value: str) -> str:
    """Return an ISO date, date-time, or year-month, or "" if untrustworthy.

    A year-month is a first-class answer rather than something to be padded to
    a first-of-month: the month is genuinely all the document gave, and saying
    so is more useful than inventing a day that reads like a stated one.
    """
    text = str(value or "").strip().replace("Z", "")
    if not text or text.lower() in {"null", "none", "unknown", "n/a", "n/a."}:
        return ""
    if DATE_RE.match(text):
        return text.replace(" ", "T", 1)
    if re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", text):
        return text
    loose = _LOOSE_DT.match(text)
    if loose:
        day, hh, mm = loose.group(1), int(loose.group(2)), int(loose.group(3))
        if hh <= 23 and mm <= 59:
            return f"{day}T{hh:02d}:{mm:02d}"
        return day
    # A bare date hiding inside something longer is still usable.
    bare = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if bare:
        return bare.group(0)
    month = re.search(r"\d{4}-(?:0[1-9]|1[0-2])(?!-?\d)", text)
    if month:
        return month.group(0)
    named = _MONTH_YEAR.search(text)
    if named:
        return f"{named.group(2)}-{_MONTHS[named.group(1)[:3].lower()]:02d}"
    return ""


# "since", "until" and their kin are the English prepositions that take a time
# complement and nothing else, so a predicate ending in one has left its
# complement unsaid - and any date attached to the assertion is a boundary on
# the fact rather than the day it happened. Rendered subject-predicate-object
# such a row reads as nonsense ("was in the section since Night Shift")
# and, worse, puts a date on the timeline that nothing happened on. "by" and
# "from" are deliberately excluded: a trailing "by" or "from" is agentive
# ("was issued by", "borrowed from") and nulling those dates would be a
# regression.
_BOUNDARY_TAIL = re.compile(
    r"(?:^|\s)(?:since|until|till|as of|up to|up until|prior to)$", re.I)


def spelled_days(text: str) -> set[int]:
    """Day numbers the quote writes in words rather than in digits."""
    out: set[int] = set()
    for m in _DAY_WORD.finditer(text):
        if m.group(1):
            tens = 20 if m.group(1).lower() == "twenty" else 30
            out.add(tens + _ORDINAL_WORDS[m.group(2).lower()])
        else:
            out.add(_ORDINAL_WORDS[m.group(3).lower()])
    return out


def date_precision(event_date: str, basis: str, quote: str) -> tuple[str, str]:
    """Make the recorded precision match the citation that supports it.

    A quote is already required to support the fact; the same evidence has to
    support the date's sharpness. A month with no day is a month - written as a
    first of the month with basis "stated" it becomes a specific day the
    document never gave, and a reader has no way to tell it from a real one.
    The same for a time: T00:00 on a page that names no clock time is a
    midnight nobody testified to.
    """
    if not event_date:
        return "", ""
    text = str(quote or "")
    # A year is four digits and so is a military time; strip years before
    # looking for a clock so "August 2026" cannot be read as 20:26.
    clock_text = _YEAR.sub(" ", text)
    day = event_date[8:10]
    if "T" in event_date and not _CLOCK.search(clock_text):
        event_date = event_date[:10]
    has_clock = "T" in event_date

    if len(event_date) == 7:
        return event_date, "approx" if _HEDGE.search(text) else "month"
    if has_clock:
        return event_date, basis if basis in ("stated", "inferred") else "stated"

    days = {int(m.group(1)) for m in _DAY_BEFORE.finditer(text)}
    days |= {int(m.group(1)) for m in _DAY_AFTER.finditer(text)}
    days |= {int(m.group(1)) for m in _DAY_ORDINAL.finditer(text)}
    days |= spelled_days(text)
    for m in _DAY_NUMERIC.finditer(text):
        days |= {int(m.group(1)), int(m.group(2))}   # D/M and M/D both
    if event_date[:10] in text:
        return event_date, "stated"
    if day and int(day) in days:
        return event_date, "approx" if _HEDGE.search(text) else "stated"
    if ((_MONTH_NAME.search(text) or re.search(r"\b\d{4}-\d{2}\b", text))
            and not days):
        # The citation names the month and no day at all, so a month is all it
        # carries. A quote that does name a day - in digits or in words - but
        # not this one is a different matter: the day came from elsewhere and
        # is inference, and truncating it to a month would throw away a
        # precision the record has while claiming the month was all there was.
        return event_date[:7], "approx" if _HEDGE.search(text) else "month"
    # The day came from somewhere else on the page, which is inference: this
    # quote does not state it, and recording it as "stated" overstates it.
    return event_date, "inferred"
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


# Rank is a closed set in this domain, and a phrase carrying one is a
# person rather than a job title.
_RANK = re.compile(
    r"\b(?:msgt|ssgt|tsgt|smsgt|cmsgt|sra|amn|a1c|capt|maj|lt|col|gen|"
    r"sgt|mr|mrs|ms|dr|det|ofc|sfc|cpl|pvt)\b\.?", re.I)

# The English first- and second-person pronouns: a closed class, listed once,
# and the same list in any document. "I" is matched case-sensitively, and the
# two shapes that wear the same letter - a middle initial ("Dana I. Alvarez")
# and a Roman-numeral label ("Section I") - are separated from the pronoun by
# _bare_i_is_pronoun below rather than by the pattern, because that decision
# needs the words on either side. "us" is lowercase only, so that an
# organisation written "US Air Force" is not read as a pronoun.
# One pattern for all singular first person, so that a rewrite walks the
# sentence in the order it is written. "I" is captured together with the
# auxiliary that follows it, because moving a sentence from first person to
# third also moves its verb agreement: "I am" is "he is", "I have" is "he has".
# Those auxiliaries are the only English verbs that inflect here; a contraction
# is the same pronoun with the verb attached, hidden from the word-boundary
# test by its apostrophe.
_FIRST_PERSON = re.compile(
    r"(?<![\w'\u2019])(?:"
    r"I['\u2019](?P<short>m|ve|ll|d)\b"
    r"|(?P<mine>(?i:me|my|myself))(?![\w'\u2019])"
    r"|I(?![\w'\u2019])(?P<aux>[ \t]+(?:haven['\u2019]t|have|don['\u2019]t|do|am)\b)?"
    r")")
# The heads that take a Roman numeral as a label. This is a closed class of
# document-structure nouns rather than case vocabulary - no investigation
# supplies it and none can exhaust it - and without it "assigned to Section I"
# is read as the pronoun and the rewrite below puts a person's name inside a
# section number.
_DESIGNATOR_HEAD = re.compile(
    r"\b(?:section|subsection|paragraph|part|phase|annex|appendix|attachment|"
    r"enclosure|exhibit|tab|chapter|article|volume|figure|table|item|line|"
    r"class|type|group|category|level|tier|building|room|wing|zone|gate|"
    r"stage|step|schedule|title)\s*$", re.I)
_AGREEMENT = {"am": "is", "have": "has", "havent": "has not",
              "do": "does", "dont": "does not"}
_SHORT_EXPANSION = {"m": "is", "ve": "has", "ll": "will", "d": "would"}
_FIRST_PLURAL = re.compile(r"(?<![\w'])(?:we|our|ours|ourselves)(?![\w'])", re.I)
_FIRST_US = re.compile(r"(?<![\w'])us(?![\w'])")
_SECOND = re.compile(r"(?<![\w'])(?:you|your|yours|yourself)(?![\w'])", re.I)


def _bare_i_is_pronoun(text: str, start: int) -> bool:
    """True when a standalone capital I at this offset is the pronoun.

    Two other things are written "I". A Roman-numeral label follows a
    document-structure noun, so the word in front settles it. A middle initial
    carries a full stop, and so does a sentence-final pronoun ("and so was
    I."); there the word in front settles it again, because an initial follows
    a given name and a pronoun follows a verb or a conjunction. Refusing every
    "I." was the safer half of that trade and it cost the sentence-final
    pronoun, which is exactly the position an unresolved "me" ends up reading
    as a second person present at the event.
    """
    before = text[:start].rstrip()
    if _DESIGNATOR_HEAD.search(before):
        return False
    previous = before.split()[-1] if before.split() else ""
    if text[start + 1:start + 2] == ".":
        return bool(previous) and not previous[0].isupper()
    return True


def _name_cased(token: str) -> bool:
    """True when a token is capitalised the way a name token is.

    One leading capital and no run of capitals after it. A word in an all-caps
    line is not evidence of anything - every word there is capitalised - so it
    does not qualify, while a single initial ("J.") and "McDonald" and
    "O'Brien" all do.
    """
    core = str(token or "").strip(".,;:!?\"'()[]\u201c\u201d\u2018\u2019")
    return bool(core) and core[0].isupper() and not core[1:].isupper()


def _has_definite_first_person(text: str) -> bool:
    """True when the text carries a first-person word no name is written with.

    Deliberately does not consult the capitalised "My"/"Me" being decided, so
    it can be asked about the company that token keeps without circling back
    on itself. A contraction, an inflected "I", a lowercase "my" or "me", and
    the plural forms are all shapes no personal name takes, so any one of them
    says the text is speaking in the first person.
    """
    for m in _FIRST_PERSON.finditer(text):
        if m.group("short") or m.group("aux"):
            return True
        if m.group("mine"):
            if not m.group("mine")[0].isupper():
                return True
        elif _bare_i_is_pronoun(text, m.start()):
            return True
    return bool(_FIRST_PLURAL.search(text) or _FIRST_US.search(text))


def _mine_token_role(text: str, start: int, end: int) -> str:
    """Whether a capitalised "My"/"Me" is the pronoun, a name token, or undecided.

    English writes the head of a possessive as a common noun, so "My office" is
    the pronoun and "My Nguyen" - or "Nguyen Thi My" - is a name, and the token
    beside it is what tells them apart. That reading needs a page whose casing
    means something. A scanned sworn statement set in capitals, and a
    title-cased OCR heading, capitalise every word they contain, so a capital
    there is not evidence of a name; reading it as one switched the first-person
    guard off for the whole page and let "MY OFFICE" through as a location that
    every interviewee then shared. Scanned and form documents are frequently
    all-caps, so that is ordinary corpus material rather than a corner case.

    Where the casing settles nothing, two things can still settle it, and
    otherwise the honest answer is "unknown". A "my" with nothing after it is
    not the possessive determiner, because English always gives that determiner
    a head noun to possess - which keeps a name ending in the token, as the
    family-name-first order writes one, readable in capitals too. And a text
    that says "I" or "we" or a lowercase "my" elsewhere is speaking in the
    first person, which no name does.

    "unknown" is returned rather than guessed at, and the caller treats it as
    unresolved without rewriting it: the guard fires, so no shared node is
    minted, and the token is left exactly as the page wrote it, so a real name
    is never overwritten with somebody else's. It is a record that this text
    could not be decided, not a decision.
    """
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    previous = before.split()[-1] if before.split() else ""
    following = after.split()[0] if after.split() else ""
    word = text[start:end]
    if (word.lower() == "my" and not following
            and previous and previous[:1].isupper()):
        return "name"
    sentence_break = before.endswith((".", "!", "?", ",", ":", ";"))
    if not ((previous[:1].isupper() and not sentence_break)
            or following[:1].isupper()):
        return "pronoun"
    if any(token[:1].islower() for token in text.split()):
        # The casing of this text is a choice, so it can be read. A name token
        # is cased like a name and the match itself is in title case; an
        # ALL-CAPS neighbour on a line that has lowercase words elsewhere is
        # emphasis or a heading, not a surname.
        if word != word.capitalize():
            return "pronoun"
        if ((_name_cased(previous) and not sentence_break)
                or _name_cased(following)):
            return "name"
        return "pronoun"
    return "pronoun" if _has_definite_first_person(text) else "unknown"


def _role_in_context(text: str, start: int, end: int, context: str) -> str:
    """Ask _mine_token_role about this token again, where the page writes it.

    A name slot holds two or three tokens and is title-cased throughout, so it
    has no lowercase word for _mine_token_role to read the casing against, and
    the honest answer about a capitalised "My"/"Me" inside one is always
    "unknown". That undecided answer was then treated as a pronoun by every
    caller that asks whether anything is unresolved, which cost a real person
    named with a leading My or Me their PERSON node and any joining of their
    conduct across documents - the harm this module's own docstring exists to
    prevent - while a name ending in the token was already rescued by the
    trailing-token branch. The slot is not the only text available: the page it
    was copied off is ordinary prose whose casing does mean something, and the
    same rules read it there with no new vocabulary.

    Occurrences are read left to right and "name" wins over "pronoun" when a
    page manages to say both, because writing a name over a token that may be
    part of somebody else's name is the error this file exists to prevent,
    while leaving a name alone costs only an unresolved word that the caller
    still logs. A page that does not contain the slot verbatim, or that reaches
    "unknown" too, claims nothing and leaves the token undecided.
    """
    if not context or not text:
        return "unknown"
    answer = "unknown"
    at = context.find(text)
    while at >= 0:
        role = _mine_token_role(context, at + start, at + end)
        if role == "name":
            return "name"
        if role == "pronoun":
            answer = "pronoun"
        at = context.find(text, at + 1)
    return answer


def _first_person_matches(text: str, include_unknown: bool = True,
                          context: str = ""):
    """The _FIRST_PERSON matches that are really first-person pronouns.

    A match whose role could not be decided is included when the caller is
    asking whether anything is unresolved, and excluded when the caller is
    about to write a name over it. `context` is the surrounding page, consulted
    only when the text on its own settles nothing.
    """
    for m in _FIRST_PERSON.finditer(text):
        if m.group("short") or m.group("aux"):
            yield m
        elif m.group("mine"):
            if m.group("mine")[0].isupper():
                role = _mine_token_role(text, m.start("mine"), m.end("mine"))
                if role == "unknown":
                    role = _role_in_context(text, m.start("mine"),
                                            m.end("mine"), context)
                if role == "name" or (role == "unknown" and not include_unknown):
                    continue
            yield m
        elif _bare_i_is_pronoun(text, m.start()):
            yield m


def has_pronoun(text: str, context: str = "") -> bool:
    """True when a name still contains a bare first- or second-person word.

    `context` is the page the name was copied off, and is optional: callers
    that have no page still get exactly the answer they got before.
    """
    value = str(text or "")
    return bool(next(_first_person_matches(value, context=context), None)
                or _FIRST_PLURAL.search(value)
                or _FIRST_US.search(value) or _SECOND.search(value))


def article_role(name: str, page_text: str, header: str) -> bool:
    """True when the document refers to this name with a definite article.

    Titles take articles and names do not: a page says "the Investigating
    Officer" and "the flight chief", never "the Wren Hargrove". This reads the
    document's own grammar instead of consulting a vocabulary list, so it
    catches roles in domains whose words were never anticipated - which is
    exactly where a blacklist fails.
    """
    text = str(name or "").strip()
    if not text or _HONORIFIC.match(text):
        return False
    # A name with no capital letter is not a name in English. "everybody",
    # "the crew", "someone" - all reach here as PERSON candidates and none of
    # them is a person.
    if not any(ch.isupper() for ch in text):
        return True

    # The head of a person's name is capitalised: "MSgt Hargrove", "de la Cruz".
    # When the last word is lowercase the phrase is a common noun with a name
    # attached to it - "Quill's neighbors", "Hargrove's crew" - which names a group
    # around the person, not the person.
    head = text.split()[-1] if text.split() else text
    if head and head[0].islower():
        return True

    blob = _WS.sub(" ", f"{page_text} {header}")
    lower = blob.lower()
    bare = text.lower()

    # An appositive after a name is a title: "Capt A. R. Nakamura, Investigating
    # Officer". The person is the name before the comma; the phrase after it
    # says what they do.
    # Guarded: an attendance line ("Present: Capt Nakamura, MSgt Hargrove") puts a
    # real person after a comma too. A title is multi-word, carries no rank,
    # and has no initial - so those three conditions separate "Nakamura,
    # Investigating Officer" from "Nakamura, MSgt Hargrove" without a list of job
    # titles. Wrongly discarding a real person is worse than keeping a phantom.
    if (len(text.split()) >= 2
            and not _RANK.search(text)
            and not re.search(r"\b[A-Z]\.", text)
            and re.search(r"[A-Z][a-z]+\.?\s+[A-Z][\w.]*[^,\n]*,\s*"
                          + re.escape(text), blob)):
        return True

    with_article = sum(lower.count(f"{a} {bare}") for a in ("the", "a", "an"))
    if not with_article:
        return False
    # Count mentions that are NOT preceded by an article; a name used bare even
    # once is being used as a name.
    total = lower.count(bare)
    return with_article >= total


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


# A blank waiting to be filled in by hand: a rule of underscores, a leader of
# dots, or an empty box. Every genre of form draws them this way and running
# prose never does, so this recognises form furniture without knowing what the
# form is for.
_BLANK_FIELD = re.compile(r"_{3,}|\.{4,}|\[\s*\]|\(\s{3,}\)")


def form_label_only(name: str, page_text: str, header: str) -> bool:
    """True when every mention of a name sits inside a blank form field.

    A signature block reads "Investigating Officer Signature: ____ Date: ____".
    Those words are the form's own furniture - the label on a line waiting for
    ink - and not a participant in the events, but no grammatical test catches
    them because the line has no grammar. The test is deliberately "only ever
    in furniture": real people sign forms too, and a real person's name also
    appears in the narrative, so a single mention outside a blank field is
    enough to keep them.
    """
    flat = _WS.sub(" ", str(name or "")).strip().casefold()
    if not flat:
        return False
    found = False
    for line in f"{page_text}\n{header}".splitlines():
        if flat not in _WS.sub(" ", line).casefold():
            continue
        found = True
        if not _BLANK_FIELD.search(line):
            return False
    return found


# Transcripts and sworn statements carry the speaker in labelled front matter.
# The label set is closed and belongs to the document genre rather than to any
# case - the same justification as the rank list above - and "interviewee" is
# tried first because a memorandum's "Subject:" line names a topic, not a
# person. A label only counts when it is used as a form label: followed by a
# colon, a full stop, or standing alone on its line. That keeps "Name of
# Investigating Officer" from being read as the value of a "Name" field.
_SPEAKER_LABELS = ("interviewee", "deponent", "affiant", "statement of",
                   "person interviewed", "subject", "witness", "name")
# "Subject:" is the one label in official correspondence that names a topic as
# often as a person, so a candidate found under it has to look like a person in
# its own right: carry a rank or honorific, or a middle initial. A topic line
# ("Subject: Household Goods Move Inventory") carries neither.
_TOPICAL_LABELS = ("subject",)
# English function words. A personal name is a sequence of proper nouns and, in
# some languages, lowercase particles ("de", "van der") - none of which are
# below. A phrase carrying one of these between its tokens is a description of
# something ("Investigation of A. Miller", "Improper conduct by Dr. Vance")
# rather than a name, however many ranks and initials it happens to contain.
# Matched lowercase as written, because a name capitalises every token it has:
# that is what keeps the "a" here from colliding with a middle initial written
# without its full stop.
_FUNCTION_WORD = re.compile(
    r"(?:^|\s)(?:of|for|by|against|regarding|re|about|concerning|in|on|at|and|"
    r"or|the|a|an|to|with|from|per|via)(?=\s)")
_INITIAL = re.compile(r"\b[A-Z]\.")
# Two to five tokens, each of which must contain a letter. Digits are allowed
# inside a token because ranks carry them ("A1C", "2Lt"), but a token that is
# only digits makes the phrase a date or a room number rather than a name, and
# a slash or a comma is punctuation no name uses.
_NAME_TOKEN = r"[A-Za-z0-9.'\-]*[A-Za-z][A-Za-z0-9.'\-]*"
_NAME_SHAPE = re.compile(rf"{_NAME_TOKEN}(?: {_NAME_TOKEN}){{1,4}}")


def strip_appositive_title(name: str) -> str:
    """Drop a trailing ", <title>" so one person is not two nodes.

    "SSgt J. Miller" and "SSgt J. Miller, Flight Chief" are the same person
    written twice, and left alone they are two graph nodes that each collect
    half of what that person said. The tail is recognised by structure rather
    than by a list of titles: it goes only when the part in front already
    designates a person on its own - two or more tokens carrying a rank, an
    honorific, or a middle initial - and the tail designates nobody, carrying
    no rank, honorific, or initial of its own. That is what separates "Miller,
    Flight Chief" from an attendance line's "Miller, SSgt Vance", and it leaves
    a reversed "Smith, John" alone because "Smith" is one token.
    """
    text = str(name or "").strip()
    while True:
        head, sep, tail = text.rpartition(",")
        head, tail = head.strip(), tail.strip()
        if not sep or not head or not tail:
            return text
        if len(head.split()) < 2 or not (
                _HONORIFIC.match(head) or _RANK.search(head)
                or _INITIAL.search(head)):
            return text
        if (_HONORIFIC.match(tail) or _RANK.search(tail)
                or _INITIAL.search(tail)):
            return text
        text = head


def sentence_shaped(value: str) -> bool:
    """True when a name slot holds a sentence rather than a name.

    The model sometimes puts a whole quoted statement where a name belongs -
    the false official statement at the centre of an allegation arrives as the
    object of a "stated" assertion. A name is a short noun phrase and carries
    no pronoun; a sentence is longer, or ends in a full stop, or says "I". All
    three tests are shape, not vocabulary. The five-token floor matters: it
    keeps a bare "me" or "my office" out of here, because those are pronouns to
    be repaired in place rather than sentences to be re-typed.

    The name-shape escape exists so that a real two-to-five-token name is never
    re-typed, and it does not apply to a phrase carrying a pronoun: no name
    contains one, so "I saw Nakamura sign it" is five name-shaped tokens and
    still a sentence. Without that exception it stayed a PERSON, and a PERSON
    is held to grounding tests a sentence cannot pass, so the row was dropped
    with its quote.
    """
    text = strip_appositive_title(
        str(value or "").strip().strip('"\u201c\u201d\u2018\u2019'))
    tokens = text.split()
    if len(tokens) < 5:
        return False
    if _NAME_SHAPE.fullmatch(text) and not has_pronoun(text):
        return False
    return bool(has_pronoun(text) or len(tokens) > 8 or text[-1] in ".!?")


_DATE_LABELS = ("date / time / location", "date/time/location", "date and time",
                "date of interview", "date of statement", "date taken", "date")

_HEADER_DATE = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\w*\.?,?\s+(\d{{4}})\b", re.I)
_HEADER_TIME = re.compile(r"\b([01]\d|2[0-3])([0-5]\d)\b")


def header_value(front_text: str, labels: tuple[str, ...]) -> str:
    """The value written against one of these labels in a form's header.

    Same shapes interviewee() has to survive: the value on the same line after
    a colon or a table pipe, or on the line below when the form renders each
    label on its own row.
    """
    for label in labels:
        pattern = re.compile(
            rf"(?mi)^{_MD_LEAD}{_MD_WRAP}{re.escape(label)}{_MD_WRAP}[ \t]*"
            rf"{_MD_WRAP}(?::|\.|\||$)[ \t]*(.*)$")
        for m in pattern.finditer(front_text):
            value = _undecorate(m.group(1) or "")
            if not value:
                rest = front_text[m.end():].lstrip("\n")
                value = _undecorate(rest.split("\n", 1)[0]) if rest else ""
            if value:
                return value
    return ""


def proceeding_datetime(front_text: str) -> str:
    """When the interview or statement itself took place, from its own header.

    An investigation's chronology is not only the conduct under investigation:
    when each witness was seen, and in what order, is part of the record and is
    routinely what shows a complaint was made after a reprimand rather than
    before. That timing is never in the prose - it sits in the form's header -
    so a prose-only extractor cannot see it at all, and the timeline loses
    every procedural milestone.

    Read from the labelled field rather than by hunting the page for a date,
    because the first date on the page may well be the conduct's, not the
    interview's.
    """
    value = header_value(front_text, _DATE_LABELS)
    if not value:
        return ""
    day = _HEADER_DATE.search(value)
    if not day:
        return ""
    month = _MONTHS.get(day.group(2)[:3].lower())
    if not month:
        return ""
    iso = f"{day.group(3)}-{month:02d}-{int(day.group(1)):02d}"
    # The clock time has to come from after the date, or a building number or a
    # room number reads as a time.
    clock = _HEADER_TIME.search(value[day.end():])
    return f"{iso}T{clock.group(1)}:{clock.group(2)}" if clock else iso


def interviewee(front_text: str, header: str = "") -> str:
    """The person whose first person this document speaks in, or "".

    Reads the form's own labels rather than guessing from content. A candidate
    then has to survive the same tests a name from the model would: it must be
    name-shaped, must not be a job title, must not be a phrase the document
    only ever writes with an article, and must appear verbatim in the text.
    A miss returns "", which leaves the document's pronouns exactly as they
    were - the previous behaviour - rather than putting a wrong name in them.
    """
    for label in _SPEAKER_LABELS:
        # The label, the separator and the value each carry their own
        # decoration on a page that came through markdown, and a pipe closes
        # the label cell the way a colon does, so "**Interviewee:** X",
        # "**Interviewee**: X" and "| Interviewee | X |" all have to reach the
        # same candidate as "Interviewee: X".
        pattern = re.compile(
            rf"(?mi)^{_MD_LEAD}{_MD_WRAP}{label}s?{_MD_WRAP}[ \t]*\d*[ \t]*"
            rf"{_MD_WRAP}(?::|\.|\||$)[ \t]*(.*)$")
        for m in pattern.finditer(front_text):
            value = _undecorate(m.group(1) or "")
            if not value:
                # A rendered form puts the label on its own line and the value
                # on the next one, which is what these transcripts look like.
                rest = front_text[m.end():].lstrip("\n")
                value = _undecorate(rest.split("\n", 1)[0]) if rest else ""
            candidate = _undecorate(value.split(",")[0]).rstrip(".")
            if not candidate or not _NAME_SHAPE.fullmatch(candidate):
                continue
            # A name does not contain English function words. This catches a
            # topic that happens to name somebody - "Investigation of A.
            # Nakamura" - which is otherwise name-shaped and grounded in the page.
            if _FUNCTION_WORD.search(candidate):
                continue
            if looks_like_role(candidate) or article_role(candidate, front_text,
                                                          header):
                continue
            # Under a label that names a topic as readily as a person, the
            # candidate has to open the way a person's name opens: with a rank,
            # an honorific, or an initial. Anywhere-in-the-phrase was not
            # enough, because a topic that mentions a person carries one too.
            if label in _TOPICAL_LABELS and not (
                    _HONORIFIC.match(candidate) or _RANK.match(candidate)
                    or re.match(r"[A-Z]\.", candidate)):
                continue
            if name_grounded(candidate, front_text, header):
                return candidate
    return ""


def short_name(name: str) -> str:
    """The surname a second mention of a person would use.

    English names a person in full once and by surname afterwards, and a claim
    that repeats a four-token name three times is both unnatural and long
    enough to hit the length cap that truncates a claim mid-sentence.
    """
    from .entities import RANKS
    tokens = [t for t in str(name or "").split()
              if t.lower().strip(".") not in RANKS and len(t.strip(".")) > 1]
    return tokens[-1] if tokens else str(name or "")


def name_for(who: str, seen: dict[str, int]) -> str:
    """Full name the first time a person is named, surname after that."""
    seen[who] = seen.get(who, 0) + 1
    return who if seen[who] == 1 else short_name(who)


def _sub_first_person(text: str, rewrite) -> str:
    """re.sub over the filtered matches, so a designator is left alone.

    A match whose role could not be decided is left alone as well. Writing a
    name over a token that may be part of somebody else's name is the one
    error this file exists to prevent, and leaving it costs only an unresolved
    word, which has_pronoun still reports and the caller still logs.
    """
    out: list[str] = []
    last = 0
    for m in _first_person_matches(text, include_unknown=False):
        out.append(text[last:m.start()])
        out.append(rewrite(m))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


# Speech attributed to somebody inside a quote. The whole point is the name
# BEFORE the first person token: "Morgan said you will regret questioning me"
# puts "me" in Morgan's mouth, not the interviewee's, however plainly the
# sentence sits in the interviewee's transcript.
_QUOTED_SPEECH = re.compile(
    r"\b([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})\s+"
    r"(?:said|says|told|shouted|yelled|replied|answered|remarked)\b", re.U)
# Words that start a sentence in capitals and are not names. Without this, "I
# said" and "He told me" attribute the speech to a pronoun.
_NOT_A_NAME = {
    # pronouns and determiners
    "i", "he", "she", "they", "we", "you", "it", "that", "this", "these",
    "those", "the", "a", "an", "who", "what", "when", "where", "which",
    # conjunctions and adverbs that begin a clause and get capitalised there
    "and", "but", "then", "so", "if", "as", "after", "afterward",
    "afterwards", "before", "later", "once", "while", "when", "because",
    "eventually", "finally", "meanwhile", "anyway", "also", "next",
    # collective nouns that are not a person
    "airman", "airmen", "everyone", "someone", "somebody", "nobody",
    "people", "personnel", "witnesses", "members", "leadership", "command",
}
_HEARD_SAY = re.compile(
    r"\b(?:heard|watched|saw)\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3})\s+"
    r"(?:say|saying|tell|telling|shout|shouting|yell|yelling)\b", re.U)


def quoted_speaker(text: str, grounded_in: str = "") -> str:
    """Who is speaking inside this text, when it is somebody else's words.

    resolve_person's docstring has always promised "the quoted person for a
    quotation inside it" and nothing ever implemented it, so first person inside
    reported speech resolved to whoever's transcript it appeared in. That turned
    "you will regret questioning me" - Morgan's words, aimed at Ellis, recorded
    in Duran's interview - into a threat against Duran, a person who was not
    party to it. A wrongly resolved pronoun leaves no residue, so unlike a
    failed resolution it logs nothing and reads perfectly.

    Only the FIRST attribution is taken, and only from before the pronoun, since
    a name after it is being spoken about rather than speaking.
    """
    for pattern in (_HEARD_SAY, _QUOTED_SPEECH):
        match = pattern.search(text or "")
        if not match:
            continue
        words = match.group(1).split()
        # A name sits next to the speech verb, so read back from it and stop at
        # the first word that cannot be part of one. Matching forward instead
        # swept up whatever began the clause: "Afterward SSgt Duran" and "If I"
        # were both being taken as the person speaking.
        kept: list[str] = []
        for word in reversed(words):
            if normalize(word).strip() in _NOT_A_NAME:
                break
            kept.insert(0, word)
            if len(kept) == 3:
                break
        name = " ".join(kept).strip()
        tokens = [t for t in normalize(name).split() if t]
        if not tokens or all(t in RANKS or t in _NOT_A_NAME for t in tokens):
            continue
        # And it has to be somebody the document actually names. Without this a
        # capitalised common noun - "Airmen said" - becomes a person, and a
        # referent that is not a person is worse than no referent at all.
        if grounded_in and not name_grounded(name, grounded_in, ""):
            continue
        return name
    return ""


def resolve_person(text: str, referent: str, speaker: str = "",
                   addressed: bool = False, reported: bool = False) -> str:
    """Put a name where the transcript left a pronoun.

    A CLAIM's text is the model's own label, so a bare "me" left inside one is
    read downstream as a second person present at the event - which turned two
    witness accounts that agree into a contradiction. The referent for first
    person is whoever the claim is attributed to, that is the assertion's own
    subject: the interviewee for direct testimony, the quoted person for a
    quotation inside it. "You" in an interviewer's question is the interviewee.
    Outside a question the second person points away from whoever is speaking,
    so it is only resolved when the assertion is attributed to somebody else -
    with the exception of a question put to the interviewee, where "you" is the
    interviewee however the assertion is attributed; the caller says so with
    "addressed". Plural first person is left exactly as written: a group cannot
    be resolved to one person, and inventing that resolution would be a worse
    error than the pronoun.
    """
    # The first mention of a person names them in full and later ones use the
    # surname, which is how English refers back to someone already named - and
    # which keeps a claim from growing past the length cap that would truncate
    # it mid-sentence.
    seen: dict[str, int] = {}
    if referent:
        def rewrite(m) -> str:
            if m.group("short"):
                return (f"{name_for(referent, seen)} "
                        f"{_SHORT_EXPANSION[m.group('short').lower()]}")
            if m.group("mine"):
                word = m.group("mine").lower()
                return (f"{name_for(referent, seen)}'s" if word == "my"
                        else name_for(referent, seen))
            tail = m.group("aux") or ""
            verb = tail.strip().lower().replace("'", "").replace("\u2019", "")
            if verb in _AGREEMENT:
                tail = tail[:len(tail) - len(tail.lstrip())] + _AGREEMENT[verb]
            return name_for(referent, seen) + tail

        text = _sub_first_person(text, rewrite)
    # Inside reported speech the second person is whoever the quoted speaker was
    # addressing, and nothing here knows who that was. Morgan's "you will regret
    # questioning me", recorded in Duran's interview, was aimed at Ellis;
    # resolving it to the interviewee invents a threat against the person who
    # merely overheard it - the same error on the other pronoun. Left as
    # written, which is the one honest answer available.
    if reported:
        return text
    if speaker and (addressed
                    or normalize(speaker) != normalize(referent or "")):
        text = _SECOND.sub(
            lambda m: (f"{name_for(speaker, seen)}'s"
                       if m.group(0).lower() in ("your", "yours")
                       else name_for(speaker, seen)),
            text)
    return text


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


def _squashed(text: str) -> tuple[str, list[int]]:
    """Lowercased text with whitespace runs collapsed, plus a map from each
    squashed index back to its index in the original.

    Per-character lower() rather than casefold(): casefold can return two
    characters for one, which would break the index map that is the whole point
    of this function.
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


def quote_offset(quote: str, page_text: str, hint: int = 0) -> int | None:
    """Where on the page the quote starts, or None if it cannot be placed.

    quote_supported already proves the quote is on the page but throws the
    position away, and the position is what says which allegation the testimony
    answers. None is a safe answer: the assertion is then left untagged, which
    means unrestricted rather than misfiled.
    """
    flat_text, idx = _squashed(page_text)
    flat_quote, _ = _squashed(quote)
    if len(flat_quote) < 8 or not idx:
        return None
    start_from = 0
    if hint > 0:
        # A page long enough to be chunked can repeat a phrase; searching from
        # where this chunk begins resolves it to the copy the model read.
        for k, orig in enumerate(idx):
            if orig >= hint:
                start_from = k
                break
    pos = flat_text.find(flat_quote, start_from)
    if pos < 0 and start_from:
        pos = flat_text.find(flat_quote)
    if pos < 0 and len(flat_quote) > 40:
        pos = flat_text.find(flat_quote[:40], start_from)
    return idx[pos] if 0 <= pos < len(idx) else None


def _anchor_number(match) -> str:
    raw = (match.group(1) or "").lower()
    if not raw:
        return ""
    return str(_ROMAN[raw]) if raw in _ROMAN else str(int(raw))


def _anchor_is_heading(text: str, match: re.Match) -> bool:
    """True when an allegation marker opens a section rather than a sentence.

    OCR wraps a sentence at the page width, so the start of a line is not the
    start of a sentence: "...I already told the IO about\nAllegation 2 and he
    moved on." puts a marker at a line start in the middle of an answer, and
    routing the rest of the page from there withholds that testimony from the
    allegation it actually answers. Two structural properties separate a
    heading from a wrapped line, and both are required. It starts a sentence:
    the line above it does not run on into it, which _line_is_open decides.
    And it introduces rather than continues: it either ends its own line or is
    closed by the punctuation that separates a heading from the text it heads.
    Both tests read the line with its markdown decoration stripped off, so that
    a bolded heading and a plain one are the same heading. Neither test knows
    anything about this case; a failure leaves the assertion untagged, which is
    unrestricted rather than misfiled.
    """
    before = text[:match.start()].rstrip(" \t")
    if before:
        if not before.endswith("\n"):
            return False
        if _line_is_open(before.rstrip("\n").rsplit("\n", 1)[-1]):
            return False
    matched = re.sub(rf"[\s{_MD_MARKS}]+$", "", match.group(0))
    if matched and matched[-1] in ".:)|\u2013\u2014-":
        return True
    rest = re.sub(rf"^[ \t{_MD_MARKS}]+", "", text[match.end():])
    return rest[:1] in ("", "\n")


def allegation_spans(text: str) -> list[tuple[int, int, str]]:
    """The character range each numbered-allegation marker governs.

    A marker owns the text from itself to the next marker or the end of the
    page. Text BEFORE the first marker is deliberately in no span: it is
    preamble, and guessing which allegation preamble belongs to is the error
    this whole mechanism exists to prevent. A page with no markers yields no
    spans, which is the correct answer for a witness interview that was never
    organised by allegation.
    """
    marks = [(m.start(), _anchor_number(m))
             for m in _ALLEGATION_ANCHOR.finditer(text)
             if _anchor_is_heading(text, m)]
    marks = [(pos, ref) for pos, ref in marks if ref]
    return [(pos, marks[i + 1][0] if i + 1 < len(marks) else len(text), ref)
            for i, (pos, ref) in enumerate(marks)]


def allegation_at(spans: list[tuple[int, int, str]],
                  offset: int | None) -> str | None:
    if offset is None:
        return None
    for start, end, ref in spans:
        if start <= offset < end:
            return ref
    return None


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
    # The model is markdown-trained and will wrap its JSON in a fenced block,
    # sometimes with a sentence in front of the fence and sometimes inside a
    # blockquote. A fence anywhere in the reply is therefore taken as the
    # payload, not only one at the very start. The closing fence is optional:
    # a reply cut off at the token cap never writes one, and its finished
    # assertions are still recoverable from what did arrive.
    text = (raw or "").strip()
    fenced = re.search(
        r"(?m)^[ \t>]*(`{3,}|~{3,})[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?"
        r"(.*?)(?:\r?\n[ \t>]*\1|\Z)", text, re.S)
    if fenced:
        text = fenced.group(2).strip()
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
    if str(entity_type).strip().upper() == "PERSON":
        # A trailing title is an appositive, not part of the name, and leaving
        # it on mints a second node for a person the graph already has.
        name = strip_appositive_title(name)
    return name[:200]


# An interviewer's question, however the transcript or the model decorates the
# line. The tag is what says the second person in this quote addresses the
# interviewee rather than pointing away from the speaker. Built from the shared
# decoration pieces, so "**Q:**", "**Q**:", "1. Q:" and "| Q: |" are all the
# same tag as "Q:".
_QUESTION_TAG = re.compile(
    rf"^\s*{_MD_LEAD}{_MD_WRAP}(?:q|io|question|interviewer)"
    rf"\s*{_MD_WRAP}\s*[:.]", re.I)


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    """Split text into sentences without breaking after a name's own full stop.

    "Capt J. M. Vance" and "Ms. Miller" each carry a full stop that ends no
    sentence, and cutting there would move a person's rank and given name out
    of the citation that names them. An initial is one letter; the rest are the
    rank and honorific abbreviations already listed above, so this needs no
    vocabulary of its own.
    """
    parts: list[str] = []
    start = 0
    for m in _SENTENCE_BREAK.finditer(text):
        word = re.search(r"([A-Za-z]+)\.$", text[:m.start()])
        token = word.group(1) if word else ""
        if token and (len(token) == 1 or _RANK.fullmatch(token)
                      or _HONORIFIC.match(token)):
            continue
        parts.append(text[start:m.start()])
        start = m.end()
    parts.append(text[start:])
    return [p for p in parts if p.strip()]


def narrow_quote(quote: str, name: str) -> str:
    """Trim a citation to the sentence that carries the fact it cites.

    Asked about a long answer, the model quotes from the start of that answer
    every time, so several assertions drawn from one answer arrive with quotes
    that are growing prefixes of one another. Downstream those read as several
    independent citations of the same words, which inflates whatever counts
    supporting evidence. Trimming each quote to the sentence that contains its
    own object leaves distinct citations and loses nothing: the sentence is
    still verbatim page text, and a quote whose object cannot be located inside
    it is returned untouched.
    """
    text = str(quote or "")
    pieces = sentences(text)
    target = _flat(name)
    if len(pieces) < 2 or len(target) < 8:
        return text
    # Shortest window first, so the sentence that carries the object wins over
    # the whole answer that also contains it.
    for width in range(1, len(pieces) + 1):
        for i in range(len(pieces) - width + 1):
            span = " ".join(pieces[i:i + width])
            if target in _flat(span):
                return span if len(span) >= 8 else text
    return text


_SERIES_DATES = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\w*\.?(?:,?\s+(\d{{4}}))?\b",
    re.I)
# An enumeration that excludes somebody must not be spread across all of its
# dates: "he was at the first but not the second" names two occasions and
# attributes one. Splitting on a sentence like this manufactures an attendance
# the sentence denies, which is the worst error available in this domain.
_EXCLUSION = re.compile(
    r"\b(?:not|never|except|excluding|but|only|missed|absent|declined|"
    r"neither|nor|without)\b", re.I)


def expand_dated_series(items: list[dict], page_text: str) -> list[dict]:
    """Turn one assertion carrying several dates into one assertion per date.

    A ledger line, a duty roster or a log extract writes a whole series in a
    single sentence - "four advances, dated 6 June, 20 June, 3 July and 17
    July". There is one event_date field, so the model leaves it empty and the
    four transactions reach the timeline as a single undated summary. Each of
    those dates is a separate event that happened on a separate day, and a
    chronology that cannot see them cannot show a pattern.

    The year is usually stated once for the whole series and omitted from the
    individual entries. Taking it from the page is not a guess when the page
    names exactly one year; where the page is ambiguous, nothing is emitted,
    because inventing a year is worse than leaving a gap.
    """
    out: list[dict] = []
    page_years = set(re.findall(r"\b(20\d{2})\b", page_text))
    for item in items:
        out.append(item)
        if item.get("event_date"):
            continue
        quote = str(item.get("quote") or "")
        if _EXCLUSION.search(quote):
            continue
        found = _SERIES_DATES.findall(quote)
        if len(found) < 2:
            continue
        dates = []
        for day, month, year in found:
            mon = _MONTHS.get(month[:3].lower())
            if not mon:
                continue
            if not year:
                if len(page_years) != 1:
                    continue
                year = next(iter(page_years))
            dates.append(f"{year}-{mon:02d}-{int(day):02d}")
        dates = sorted(set(dates))
        if len(dates) < 2:
            continue
        # The undated summary is replaced by the series it describes; keeping
        # both would leave one dateless assertion standing for events that now
        # have their own dates, and every count of them would be one too many.
        out.pop()
        for when in dates:
            copy = dict(item)
            copy["event_date"] = when
            # The day and month are the document's own words; the year came
            # from elsewhere on the page, so this is resolved, not stated.
            copy["event_date_basis"] = "inferred" if not found[0][2] else "stated"
            out.append(copy)
    return out


def dedupe_assertions(items: list[dict]) -> list[dict]:
    """Drop assertions that say the same thing off the same words.

    A page long enough to be chunked, or an answer the model reads twice, can
    yield the same subject, predicate and object twice with one quote contained
    in the other. Counted separately they read as two witnesses to one fact,
    and any weighing that counts supporting evidence is then counting the same
    sentence twice. The longer quote survives, because it contains the shorter
    one and is the more informative citation. Assertions whose objects differ
    are never folded together: a distinct object is a distinct fact, and losing
    one would cost more than the double count.
    """
    kept: list[dict] = []
    seen: list[tuple[tuple, str]] = []
    for item in sorted(items, key=lambda i: -len(str(i.get("quote") or ""))):
        # The event date is part of what an assertion says. One sentence
        # listing four dates for the same conduct is meant to become four
        # assertions - same subject, predicate and object, different day - and
        # folding on the identity alone silently discarded three of them, so a
        # ledger of dated transactions collapsed to a single undated summary.
        key = (item["subject_type"], _flat(item["subject_name"]),
               item["predicate"], item["object_type"],
               _flat(item["object_name"]), str(item.get("event_date") or ""))
        flat = _flat(str(item.get("quote") or ""))
        if any(key == other and (flat in text or text in flat)
               for other, text in seen):
            continue
        seen.append((key, flat))
        kept.append(item)
    return kept


def _name_fault(name: str, etype: str, page_text: str, header: str,
                as_written: str = "") -> str:
    """Why this PERSON or ORG name cannot stand as a name, or "" if it can.

    The four tests live in one place so that the caller decides what a fault
    costs, rather than each test deciding for itself that the whole assertion
    dies. `as_written` is the same slot before any pronoun in it was resolved:
    grounding accepts either spelling, because the repair puts a name this
    document itself supplies where the page left "I", and grounding is a test
    of the page's own words rather than of the pipeline's rewriting of them.
    """
    text = str(name or "")
    if not name_grounded(text, page_text, header) and not (
            as_written and name_grounded(str(as_written), page_text, header)):
        return "does not appear in the document"
    if etype != "PERSON":
        return ""
    if looks_like_role(text):
        return "is a job title, not a person"
    if article_role(text, page_text, header):
        return ("is only ever written with an article, so it is a role rather "
                "than a name")
    # A label on a form is not a participant. The page writes it only where a
    # blank waits for a signature, so nothing in the record has it doing or
    # saying anything.
    if form_label_only(text, page_text, header):
        return ("appears only as a form label beside a blank, so it names no "
                "participant")
    return ""


def validate(item: dict, page_text: str, header: str = "",
             speaker: str = "") -> tuple[dict | None, str]:
    for field in ("subject_type", "subject_name", "predicate",
                  "object_type", "object_name", "quote"):
        if not str(item.get(field) or "").strip():
            return None, f"missing {field}"

    # The citation is settled before the names are, because both of the steps
    # below need it - the repair reads the question tag off the front of it -
    # and because an assertion whose quote is not on the page is going nowhere
    # whatever its names say. Narrowing still happens here, while the object is
    # written the way the page writes it and can be found inside the quote.
    quote = str(item["quote"]).strip()[:MAX_QUOTE_CHARS]
    if not quote_supported(quote, page_text):
        return None, "quote does not appear on the page"
    quote = narrow_quote(quote, str(item["object_name"]))

    # A whole quoted sentence in a name slot is not a bad name, it is a
    # misfiled claim - and often the very statement an allegation turns on.
    # Dropping it loses primary evidence, so it is re-typed as the CLAIM it
    # already is and keeps its quote; the name tests below then correctly leave
    # it alone.
    for role, type_field in (("subject_name", "subject_type"),
                             ("object_name", "object_type")):
        if (str(item.get(type_field) or "").strip().upper() in ("PERSON", "ORG")
                and sentence_shaped(str(item[role]))):
            log.info("re-typed %s as CLAIM, it is a sentence not a name: %.60s",
                     role, str(item[role]))
            item[type_field] = "CLAIM"

    # One pronoun rule for every slot, applied before any slot can be rejected
    # for carrying one. A name slot holding "me" or "my office" is prose the
    # model mislabelled, and which of the six labels it reached for is not
    # evidence about anything: rejecting the row under DOCUMENT while repairing
    # the identical string under EVENT threw away the quote - primary evidence,
    # and a quoted statement attributed to a named person is often the most
    # important artefact on a page - on the strength of a label. So the pronoun
    # is repaired wherever it appears, and a slot the repair cannot rescue is
    # re-labelled rather than the assertion being dropped.
    #
    # The referent for first person is whoever the assertion is attributed to,
    # that is its own subject when that subject is a person, so a quotation
    # attributed to someone other than the interviewee resolves to the person
    # quoted. A subject that is itself a pronoun names nobody and cannot serve
    # as a referent - substituting "me" for "my" writes nonsense rather than a
    # name - so the speaker stands in for it.
    #
    # A name slot is title-cased throughout and so cannot settle a capitalised
    # "My" or "Me" on its own; the page and the header can, and are handed over
    # for that one question. Both are passed because a document names a person
    # in its front matter as readily as in the answer being read.
    as_written = {role: str(item[role])
                  for role in ("subject_name", "object_name")}
    around = f"{header}\n{page_text}"
    referent = (as_written["subject_name"]
                if str(item["subject_type"]).strip().upper() == "PERSON"
                and not has_pronoun(as_written["subject_name"], around)
                else speaker)
    # Reported speech overrides both. "Morgan said you will regret questioning
    # me" attributes the "me" to Morgan wherever it is recorded, and resolving
    # it to the interviewee invents a remark about a person who was listening.
    quoted = quoted_speaker(quote, around)
    if quoted and normalize(quoted) != normalize(referent or ""):
        log.info("first person inside reported speech resolves to %s, not %s",
                 quoted, referent or speaker or "the interviewee")
        referent = quoted
    addressed = bool(_QUESTION_TAG.match(quote))
    for role, type_field in (("subject_name", "subject_type"),
                             ("object_name", "object_type")):
        prose = str(item.get(type_field) or "").strip().upper() in ("CLAIM",
                                                                    "EVENT")
        if not prose and not has_pronoun(str(item[role]), around):
            continue
        item[role] = resolve_person(str(item[role]), str(referent or ""),
                                    speaker, addressed=addressed,
                                    reported=bool(quoted))
        # The repair is not guaranteed to succeed - a claim can be attributed
        # to the speaker themselves, which leaves the second person pointing at
        # somebody this page never names, and a page with no identified speaker
        # offers nothing to resolve to at all. The assertion is kept either
        # way, because the quote is evidence whatever the name slot ends up
        # holding, and the residue is logged so a run can be audited for it
        # rather than having to be re-read. Outside CLAIM and EVENT the slot is
        # also re-typed, because a slot still holding a pronoun is not the name
        # of anything: that is the same move the sentence-shaped branch above
        # makes, and it is a record that the pipeline failed to resolve a word,
        # not a finding about what the document says.
        if has_pronoun(str(item[role]), around):
            log.info("%s still carries a pronoun after repair%s: %.80s", role,
                     "" if prose else ", re-typed as CLAIM", str(item[role]))
            if not prose:
                item[type_field] = "CLAIM"

    # People and organisations must be traceable to ink on a page. CLAIM and
    # EVENT names are the model's own summarising label, so they are exempt.
    # A name test that fails re-types the slot to CLAIM and keeps the row; it
    # never discards the assertion. Two things are true at once here, and the
    # re-typing is what satisfies both: no PERSON or ORG node may be minted
    # from a job title or from a name the page never writes, and the quote is
    # verbatim page text whose survival cannot be allowed to depend on which
    # of six labels a markdown-trained model reached for. Dropping the row
    # made survival depend on exactly that - the identical string was kept
    # under LOCATION, DOCUMENT, EVENT and CLAIM and destroyed under PERSON and
    # ORG - and a quoted statement attributed to a named person is frequently
    # the single most important artefact in an investigation, so dropping it
    # is the worst outcome available. This is the same move the sentence-shaped
    # and pronoun branches above already make, applied to the one path that
    # had not been given it.
    for role, type_field in (("subject_name", "subject_type"),
                             ("object_name", "object_type")):
        etype = str(item.get(type_field) or "").strip().upper()
        if etype not in ("PERSON", "ORG"):
            continue
        fault = _name_fault(str(item[role]), etype, page_text, header,
                            as_written=as_written[role])
        if not fault:
            continue
        # Logged as a pipeline failure to name the slot, not as a finding
        # about the document: the record says this run could not read a name
        # out of these words, and says nothing about whether the underlying
        # allegation is supported. Whether the slot arrived carrying a pronoun
        # is reported because it is the difference between prose the repair
        # could not rescue and a name the page does not support, and an audit
        # of a run needs to tell those apart - but it no longer changes what
        # happens to the row.
        log.info("could not read %s as %s (%s); re-typed as CLAIM, quote "
                 "kept%s: %.60s", role, etype, fault,
                 ", it held a pronoun" if has_pronoun(as_written[role],
                                                      around) else "",
                 str(item[role]))
        item[type_field] = "CLAIM"

    # Read after every re-typing above, so that a slot re-labelled CLAIM is
    # stored as one rather than under the label it arrived with.
    #
    # A label outside the schema is a formatting slip - "PEOPLE" for PERSON, a
    # translated or pluralised word, a label the reply invented - and dropping
    # the row for it put the survival of a verbatim quote back in the hands of
    # which word the model happened to type, which is the failure every branch
    # above exists to close. The slot is re-typed to CLAIM instead: an
    # unrecognised label carries no assurance that the text is a person, an
    # organisation or a place, and CLAIM is the label for text the pipeline has
    # not verified as any of those. Nothing about the document is inferred from
    # the slip, and the citation survives it.
    for field in ("subject_type", "object_type"):
        value = str(item[field]).strip().upper()
        if value not in ENTITY_TYPES:
            log.info("%s is not a type this schema knows (%r); re-typed as "
                     "CLAIM, quote kept", field, value)
            value = "CLAIM"
        item[field] = value
    subject_type, object_type = item["subject_type"], item["object_type"]

    event_date, basis = date_precision(
        normalize_when(item.get("event_date")),
        str(item.get("event_date_basis") or "").strip().lower(), quote)
    if basis not in ("stated", "month", "approx", "inferred"):
        basis = "stated" if event_date else ""
    if not event_date:
        basis = ""
    # A predicate left ending in a temporal preposition has swallowed the
    # boundary word whose complement is the date, so the date bounds the fact
    # rather than dating it. The assertion keeps its wording and its quote;
    # only the false event date goes.
    if event_date and _BOUNDARY_TAIL.search(str(item["predicate"]).strip()):
        event_date, basis = "", ""

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


# A comma inside a PERSON name joins two forms of one name rather than
# separating two names: a page writes "Wren, SSgt Sorenson" when it restates a
# given name as the ranked form, and "Sorenson, Wren" when it inverts a name
# for a roster. Neither is a name a report can print, and the model copies
# whichever the page wrote straight into the name slot. This spots the shape
# only; it never decides who the person is, because both halves normalise into
# the same entity id either way and the id is what says which person this is. A
# trailing appositive title is already removed by _clean_name before this is
# asked, so a comma still present here is not "Miller, Flight Chief".


def name_form_priority(entity_type: str, name: str) -> tuple:
    """Sort key choosing which written form labels an entity. Lowest wins.

    One entity id collects every spelling of one name - "SSgt Wren Sorenson"
    and "Wren, SSgt Sorenson" normalise identically, which is why they are one
    node at all - and one of those spellings has to be the label a report and a
    graph show. Taking the first one seen made that label an accident of the
    order documents happened to be processed in, and one page that wrote a
    comma-joined form ahead of a dozen well-formed mentions named the person
    that way for the whole graph.

    So the form is chosen rather than inherited. A PERSON form carrying no
    comma beats one that does, because a comma there means two forms of the
    name were written together and a label has to be one name. After that the
    fuller form wins - more tokens, then more characters - which is the rule
    _merge_priority already uses to label a merged node, so a rank and a given
    name are kept where the corpus supplies them. The form itself breaks the
    last tie, so the same set of mentions always yields the same label whatever
    order they arrive in.

    This chooses a spelling; it never joins two people. The entity id is
    computed before this is asked and is not touched by it, so two people who
    share a surname stay exactly as separate after a re-label as before.
    """
    text = str(name or "").strip()
    conjoined = str(entity_type).strip().upper() == "PERSON" and "," in text
    return (int(conjoined), -len(text.split()), -len(text), text)


def register_entity(conn, entity_type: str, name: str) -> None:
    eid = entity_id(entity_type, name)
    text = name.strip()
    conn.execute(
        """INSERT INTO entities (entity_id, entity_type, canonical_name,
                                 first_seen, mention_count)
           VALUES (?,?,?,?,1)
           ON CONFLICT(entity_id) DO UPDATE SET
             mention_count = entities.mention_count + 1""",
        (eid, entity_type, text, utcnow()))
    # Read back rather than resolved in SQL: the comparison is a Python sort
    # key over both written forms, and re-labelling on the way past is what
    # makes the stored label independent of the order the mentions arrived in.
    row = conn.execute("SELECT canonical_name FROM entities WHERE entity_id=?",
                       (eid,)).fetchone()
    stored = row[0] if row else text
    if stored != text and (name_form_priority(entity_type, text)
                           < name_form_priority(entity_type, stored)):
        log.info("relabelled %s as %r; %r is the same name written another "
                 "way", eid, text, stored)
        conn.execute("UPDATE entities SET canonical_name=? WHERE entity_id=?",
                     (text, eid))


def run(doc_id: str, on_progress) -> tuple[int, int]:
    rows = state.query(
        """SELECT doc_id, page_num, text_path FROM pages
           WHERE doc_id=? AND text_path IS NOT NULL ORDER BY page_num""", (doc_id,))
    if not rows:
        return 0, 0

    client = Ollama()
    # The name has to come from the same place the endpoint did, or extraction
    # asks the operator's server for the local model's name.
    model = client.require_model(llm_settings.effective_text_model(),
                                 llm_settings.text_model_label())
    options = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                              "EXTRACT_NUM_PREDICT", 1200)

    # Page 1's opening usually carries the metadata header that names the
    # interviewee - the anchor for resolving "I" and "you" on every page.
    front = ""
    first = paths.under_root(rows[0]["text_path"]) if rows else None
    if first is not None:
        front = first.read_text(encoding="utf-8")
    header = front[:500]
    # Only a document that records somebody speaking has a first person to
    # resolve. The gate is doc_kind and not doc_role: classify_role falls back
    # to "witness" for anything it cannot place, so a role is always present
    # and gates nothing, while a memorandum or a log is a kind of its own and
    # its "Subject:" line names a topic rather than a person. The
    # 2500-character window is the one ingest reads the front matter through.
    doc = state.query_one("SELECT doc_kind FROM documents WHERE doc_id=?", (doc_id,))
    kind = (doc["doc_kind"] or "").strip().lower() if doc else ""
    speaker = (interviewee(front[:2500], header)
               if kind in ("interview", "statement") else "")
    if speaker:
        log.info("%s: first person resolves to %s", doc_id, speaker)

    kept = dropped = 0

    # The proceeding itself is a dated event in the investigation's chronology,
    # and its date is in the header rather than the prose, so no amount of
    # reading the body will produce it. Recording it is what lets a reader see
    # the order the investigation actually happened in - which witness was seen
    # before which, and whether a complaint preceded or followed the discipline
    # it is said to answer.
    when = proceeding_datetime(front) if kind in ("interview", "statement") else ""
    if when and speaker:
        label = header_value(front, _DATE_LABELS)
        quote = f"{header_value(front, _SPEAKER_LABELS) or speaker} - {label}".strip(" -")
        tid = triple_id(doc_id, rows[0]["page_num"],
                        ("PERSON", speaker, "was interviewed on",
                         "EVENT", "this interview", when))
        with state.tx() as conn:
            conn.execute(
                """INSERT INTO triples (triple_id, doc_id, page_num,
                     subject_type, subject_name, predicate, object_type,
                     object_name, event_date, event_date_basis,
                     allegation_ref, quote, model, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(triple_id) DO NOTHING""",
                (tid, doc_id, rows[0]["page_num"], "PERSON", speaker,
                 "was interviewed on", "EVENT", "this interview", when,
                 "stated", None, quote, "header", utcnow()))
            register_entity(conn, "PERSON", speaker)
        kept += 1
        log.info("%s: proceeding recorded as %s on %s", doc_id, speaker, when)
    for idx, row in enumerate(rows, 1):
        on_progress(f"extracting from page {idx}/{len(rows)} with {model}")
        text_file = paths.under_root(row["text_path"])
        if text_file is None:
            continue
        page_text = text_file.read_text(encoding="utf-8").strip()
        if not page_text or page_text == "[no text]":
            continue

        # Spans are computed over the whole page, never over a chunk, so a
        # marker in one chunk still governs the chunk after it.
        spans = allegation_spans(page_text)

        # Assertions are collected for the whole page before any are written,
        # so that two chunks quoting the same sentence can be recognised as one
        # assertion rather than stored as two.
        page_items: list[dict] = []
        for chunk in chunks(page_text):
            chunk_at = page_text.find(chunk)    # chunks are contiguous slices
            if chunk_at < 0:
                chunk_at = 0
            system, user, version = build_prompt(doc_id, row["page_num"], chunk,
                                                 header=header)
            try:
                data = client.generate(model, user, system=system, options=options,
                                       format_json=True, think=thinking_enabled())
            except Exception as exc:
                log.error("%s p%s: %s", doc_id, row["page_num"], exc)
                continue

            reply = str(data.get("response") or "")
            items = parse_response(reply)
            # A reply with substance in it that yields no assertions is a
            # parsing failure, not an empty page, and silence there would hand
            # the rest of the pipeline a page it thinks had nothing on it.
            if not items and len(reply.strip()) > 40:
                log.error("%s p%d: the reply had content but parsed to zero "
                          "assertions; this page's evidence is being lost. "
                          "First 200 characters: %s", doc_id, row["page_num"],
                          " ".join(reply.split())[:200])
            for item in items:
                clean, reason = validate(item, chunk, header=header,
                                         speaker=speaker)
                if clean is None:
                    dropped += 1
                    log.info("%s p%d: dropped (%s)", doc_id, row["page_num"], reason)
                    continue
                clean["_chunk_at"] = chunk_at
                page_items.append(clean)

        page_items = expand_dated_series(page_items, page_text)
        for clean in dedupe_assertions(page_items):
            ref = allegation_at(
                spans, quote_offset(clean["quote"], page_text,
                                    clean.pop("_chunk_at", 0)))
            # The date belongs in the identity for the same reason it belongs
            # in the dedupe key: without it, ON CONFLICT DO NOTHING keeps the
            # first of a dated series and drops the rest.
            tid = triple_id(doc_id, row["page_num"],
                            (clean["subject_type"], clean["subject_name"],
                             clean["predicate"], clean["object_type"],
                             clean["object_name"],
                             str(clean.get("event_date") or "")))
            with state.tx() as conn:
                conn.execute(
                    """INSERT INTO triples (triple_id, doc_id, page_num,
                         subject_type, subject_name, predicate, object_type,
                         object_name, event_date, event_date_basis,
                         allegation_ref, quote, model, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(triple_id) DO NOTHING""",
                    (tid, doc_id, row["page_num"], clean["subject_type"],
                     clean["subject_name"], clean["predicate"], clean["object_type"],
                     clean["object_name"], clean["event_date"],
                     clean["event_date_basis"], ref, clean["quote"],
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


def _merge_edges() -> dict[str, str]:
    """Every recorded merge, as a map from the merged entity to its target."""
    return {r["entity_id"]: r["merged_into"] for r in state.query(
        "SELECT entity_id, merged_into FROM entities WHERE merged_into IS NOT NULL")}


def _merge_root(eid: str, edges: dict[str, str]) -> str:
    """Follow a merge chain to the entity that is not itself merged."""
    seen: set[str] = set()
    current = eid
    while current in edges and edges[current] and current not in seen:
        seen.add(current)
        current = edges[current]
    return current


def _merge_priority(name: str, mention_count, eid: str) -> tuple:
    """Sort key that decides which of two entities survives a merge.

    The fuller name wins - more name tokens, then more characters - because a
    merged node has to be labelled with the form that identifies the person to
    a reader, and because the answer must not depend on which row the merge was
    discovered from. Mention count breaks a tie between two equally complete
    names, and the entity id breaks a tie between those, so the same pair
    always merges the same way however the rows are ordered. Between two
    spellings of one name, the one without a trailing title is preferred: the
    title is the appositive that made them two nodes in the first place, and it
    has no business becoming the label the graph shows.
    """
    text = str(name or "")
    norm = normalize(strip_appositive_title(text))
    return (-len(norm.split()), -len(norm),
            int(strip_appositive_title(text) != text.strip()),
            -int(mention_count or 0), eid)


def compact_merges() -> int:
    """Make the merge graph a forest: every chain ends at an unmerged root.

    Two documents can assert the same alias in opposite directions ("Miller is
    also known as Tank" on one page, "Tank is also known as Miller" on
    another). Recording both left the rows pointing at each other, and a cycle
    has no canonical entity at all: resolving either one either loops or
    returns whichever row the resolver happened to start from. Cycles are
    broken in favour of the fuller name, and every surviving chain is then
    rewritten to point straight at its root, so one lookup resolves any entity
    and no resolver has to defend itself against a loop.
    """
    rows = state.query("SELECT entity_id, canonical_name, merged_into, "
                       "mention_count FROM entities")
    info = {r["entity_id"]: r for r in rows}
    edges = {r["entity_id"]: r["merged_into"] for r in rows if r["merged_into"]}
    changes: dict[str, str | None] = {}

    # Cycles are broken first: until they are, following a chain does not end.
    settled: set[str] = set()
    for start in list(edges):
        if start in settled:
            continue
        path: list[str] = []
        walked: set[str] = set()
        current = start
        while (current in edges and current not in walked
               and current not in settled):
            walked.add(current)
            path.append(current)
            current = edges[current]
        if current in walked:
            cycle = path[path.index(current):]
            winner = min(cycle, key=lambda e: _merge_priority(
                info[e]["canonical_name"] if e in info else e,
                info[e]["mention_count"] if e in info else 0, e))
            log.warning("merge cycle over %d entities broken in favour of %s",
                        len(cycle), info[winner]["canonical_name"]
                        if winner in info else winner)
            for member in cycle:
                target = None if member == winner else winner
                changes[member] = target
                if target:
                    edges[member] = target
                else:
                    edges.pop(member, None)
        settled |= walked

    for eid in list(edges):
        root = _merge_root(eid, edges)
        if root != edges.get(eid):
            edges[eid] = root
            changes[eid] = root

    if changes:
        with state.tx() as conn:
            for eid, target in changes.items():
                conn.execute(
                    "UPDATE entities SET merged_into=? WHERE entity_id=?",
                    (target, eid))
    return len(changes)


def _record_merge(a: str, b: str, edges: dict[str, str],
                  info: dict) -> tuple[str, str] | None:
    """Merge two entities and return (kept, lost), or None if nothing to do.

    Both sides are resolved to their roots first, and the edge is always
    written from one root to another. A root has no outgoing merge, so adding
    one edge from it into a different tree cannot close a loop - which is the
    property that makes the merge graph provably acyclic rather than merely
    loop-guarded at read time.
    """
    root_a, root_b = _merge_root(a, edges), _merge_root(b, edges)
    if root_a == root_b or root_a not in info or root_b not in info:
        return None
    keep, lose = sorted((root_a, root_b), key=lambda e: _merge_priority(
        info[e]["canonical_name"], info[e]["mention_count"], e))
    with state.tx() as conn:
        conn.execute("UPDATE entities SET merged_into=? WHERE entity_id=?",
                     (keep, lose))
    edges[lose] = keep
    return keep, lose


def merge_stated_aliases() -> int:
    """Apply nicknames the documents actually assert.

    Spelling cannot reveal that Wren Hargrove is called Quill; only a sentence can.
    These merges are therefore evidence-backed rather than heuristic, and are
    applied before the similarity pass so the alias is already resolved. Which
    of the two names survives is not taken from the direction of the sentence:
    the same alias is stated both ways round across a case file, so the
    direction is an accident of who was talking, and the fuller name is chosen
    instead.
    """
    rows = state.query("SELECT entity_id, canonical_name, mention_count "
                       "FROM entities WHERE entity_type='PERSON'")
    info = {r["entity_id"]: r for r in rows}
    edges = _merge_edges()
    merged = 0
    for row in state.query(
            "SELECT subject_name, object_name FROM triples "
            "WHERE predicate LIKE '%also known as%' OR predicate LIKE '%goes by%' "
            "OR predicate LIKE '%is called%' OR predicate LIKE '%nicknamed%'"):
        full, nick = row["subject_name"], row["object_name"]
        if not full or not nick or normalize(full) == normalize(nick):
            continue
        outcome = _record_merge(entity_id("PERSON", full),
                                entity_id("PERSON", nick), edges, info)
        if not outcome:
            continue
        merged += 1
        log.info("merged stated alias %s into %s",
                 info[outcome[1]]["canonical_name"],
                 info[outcome[0]]["canonical_name"])
    return merged


def auto_merge(on_progress=lambda _m: None) -> int:
    """Fold obvious name variants together: Smith / SSgt Smith / J. Smith.

    Runs automatically.  Only high-confidence forms merge - an initialism or a
    surname against a full name, or an exact match after rank and punctuation
    are stripped.  Anything less certain is left as separate entities, because
    wrongly joining two people is worse than showing two nodes.
    """
    from itertools import combinations

    # Any cycle or chain left by an earlier pass is flattened before anything
    # is added to the graph, so every merge below starts from a real root.
    compact_merges()
    merge_stated_aliases()
    threshold = float(env_int("MERGE_THRESHOLD", 88))
    rows = state.query(
        "SELECT entity_id, entity_type, canonical_name, mention_count FROM entities "
        "WHERE merged_into IS NULL ORDER BY entity_type, mention_count DESC")
    info = {r["entity_id"]: r for r in state.query(
        "SELECT entity_id, canonical_name, mention_count FROM entities")}
    edges = _merge_edges()

    by_type: dict[str, list] = {}
    for row in rows:
        by_type.setdefault(row["entity_type"], []).append(row)

    merged = 0
    for entity_type, group in by_type.items():
        if entity_type not in {"PERSON", "ORG", "LOCATION"} or len(group) < 2:
            continue
        # Compared with the trailing title removed, so that "Miller" and
        # "Miller, Flight Chief" are one person before any similarity score is
        # asked for.
        norms = {r["entity_id"]: normalize(strip_appositive_title(
            r["canonical_name"])) for r in group}

        # A bare surname that fits two different people identifies neither.
        # "Sorenson" scores 90 against both "Anton Sorenson" and "Wren
        # Sorenson", and merging into whichever was compared first silently
        # attributes one person's conduct to the other - the worst error this
        # pipeline can make, reached by trying to be helpful.
        ambiguous: set[str] = set()
        for candidate in group:
            rivals = [other for other in group
                      if other["entity_id"] != candidate["entity_id"]
                      and _score(norms[candidate["entity_id"]],
                                 norms[other["entity_id"]]) >= threshold]
            if len(rivals) >= 2:
                ambiguous.add(candidate["entity_id"])
                log.info("%s matches %d people; left unmerged as ambiguous",
                         candidate["canonical_name"], len(rivals))

        for a, b in combinations(group, 2):
            if a["entity_id"] in ambiguous or b["entity_id"] in ambiguous:
                continue
            score = _score(norms[a["entity_id"]], norms[b["entity_id"]])
            if score < threshold:
                continue
            # Both sides are followed to their roots first: either may already
            # have been merged earlier in this same loop, and writing an edge
            # into a node that is itself merged is how a chain, or a loop,
            # gets built. The fuller name survives.
            outcome = _record_merge(a["entity_id"], b["entity_id"], edges, info)
            if not outcome:
                continue
            merged += 1
            log.info("merged %s into %s (%.0f)",
                     info[outcome[1]]["canonical_name"],
                     info[outcome[0]]["canonical_name"], score)
    # Merging a root into another root can leave a chain two long; flattening
    # it here means every entity resolves in a single lookup.
    compact_merges()
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
