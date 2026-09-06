"""Generate a report against a standing investigative objective.

The objective is a list of allegations. Each one is answered on its own, from
its own evidence, because a single prompt covering everything encourages the
model to blur one allegation's evidence into another's - which is the failure
that matters most in a document like this.

Every finding must cite the file and page it rests on, and each allegation ends
in one of two dispositions:

  Substantiated       - the evidence carries every element of the allegation
  Not substantiated   - it does not

Those are the default words for them. The doctrine is fixed but the wording is
the appointing authority's, so the pair is read from settings by
disposition_labels() and every prompt, verdict and table below is rendered from
whatever it returns. There is no third label. The burden sits on the allegation, so an allegation the
evidence does not carry is not substantiated, and what is missing is written as
a gap rather than dressed up as a disposition of its own. A third label reads as
a verdict on the investigation instead of on the claim, and it was being reached
for whenever the record was merely inconvenient.

An allegation is decomposed into its constituent elements before any evidence is
weighed, and every element has to be met for the allegation to be substantiated.
Decomposing first is what stops a single admission from carrying a claim it does
not answer: an allegation asserting a course of conduct contains an element that
one conceded instance cannot meet, however clearly that instance is conceded.

Which elements those are is the decomposition pass's own reading of the
allegation, made with no evidence in front of it, and that pass is the only
source for them. Nothing in this module decides from a list of words that an
allegation asserts repetition, and nothing decides from a list of words that it
does not: a vocabulary tuned on one investigation reads that investigation back
out of every other one, and a check that can only ever move a disposition in one
direction is not a check. Whether repetition was established is arithmetic on
the count the weighing itself states and the dated occasions carried by the
pages that weighing cites, which settles the question in either direction with
equal willingness. Where neither is readable there is nothing to do the
arithmetic on, and the element is recorded as unweighed rather than failed: a
line the model did not write is not evidence that the conduct happened once.

The same rule governs everything below that could otherwise decide an
allegation by accident. A parse failure, a citation this module cannot tie to
an ingested document, an element table that does not cover the elements it was
given - none of them is a finding about the evidence, all of them are recorded
as failures to weigh, and each one is named in the section and in the basis
column beside the disposition. The disposition of record in that case is still
that the allegation is not carried, because the burden sits on the allegation,
but it is labelled as what it is and it must never be reachable through an
ordinary variation in how the model formats its answer. That is what the single
markdown normalisation below exists for.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import chat, embed, evidence, graph, llm_settings, state
from .log import get_logger, utcnow
from .model_client import (Ollama, default_options, random_seed,
                           thinking_enabled)

log = get_logger("report")

OBJECTIVE_KEY = "cdi_objective"          # legacy free-text block
GOAL_KEY = "cdi_goal"
ALLEGATIONS_KEY = "cdi_allegations"      # JSON list, one string per allegation
MAX_PASSAGES = 10
# Routing discards passages, so retrieval has to over-fetch or the pass is
# starved by its own filter.
RETRIEVAL_WIDTH = 4
MAX_RELATIONSHIPS = 40

SUBSTANTIATED = "Substantiated"
NOT_SUBSTANTIATED = "Not substantiated"
# The default labels in one place. Every regex, prompt and stored verdict below
# is built from the pair returned by disposition_labels(), because the enum used
# to be retyped as a literal in six places and a rename would have been caught
# in one of them.
DISPOSITIONS = (SUBSTANTIATED, NOT_SUBSTANTIATED)
LABELS_KEY = "cdi_disposition_labels"    # JSON pair, the affirmative label first

# The text model is markdown-trained. It decorates structured output with bold,
# bullets, headings, numbering and whole pipe tables however plainly the shape
# was asked for, so '**E1** | ELEMENT: x', '| E1 | ELEMENT: x |',
# '**MULTIPLICITY:** **yes**' and '### Findings:' are all normal answers to the
# templates below.
#
# Every previous round of this module answered that by adding one more optional
# decoration group to whichever regex had just been caught out, and that is how
# the holes were made: each anchor ended up with its own idea of what decoration
# looks like, and each of them was wrong about a different shape - emphasis on
# both sides of a colon, emphasis on the identifier alone, a leading table pipe,
# a trailing colon after a heading. Decoration is therefore removed ONCE, in
# _Markup below, and every anchor in this module is matched against the
# decoration-free copy. A new anchor is written in its plain form and inherits
# the tolerance rather than restating it.
#
# The delivered section stays the model's own text, decoration and all, so
# _Markup keeps a map from each position of the readable copy back to the
# position of the original it came from: a correction is worked out on the copy
# and written into the original at the mapped offset.
#
# A table pipe is decoration in the same sense a bullet is, so it belongs in
# this class: it is what the model puts around its content, not the content. It
# is here rather than only in _line_decoration_end because _NUMBERED_ITEM below
# is matched against the ORIGINAL text, where '| 1. first item |' has to be the
# same numbered item as '1. first item'.
_MD_DECOR = r"(?:[>#*_+~•|-]+[ \t]*)*"
# The markers alone, without the leading indent, for the one pattern that has to
# know whether a heading sits at the margin or inside an indented block.
_MD_MARKERS = _MD_DECOR + r"(?:\d+[.)][ \t]*" + _MD_DECOR + r")?"
_MD_LEAD = r"[ \t]*" + _MD_MARKERS
# Emphasis characters carry no meaning in these answers and are dropped wherever
# they appear. Underscores are treated separately, because one inside a word is
# part of a filename ("custodian_log.pdf") and a citation that loses it stops
# naming the document it cites.
_MD_MARKS = "*`~"


def _anchored(pattern: str) -> str:
    """`pattern` at the start of a line of _Markup.text.

    The leading tolerance is kept even though _Markup has already removed the
    decoration, because a few patterns built here are still matched against raw
    text and the extra latitude costs nothing on normalised text.

    The multiline flag is left to the caller rather than embedded as an inline
    (?m), because an inline global flag is only legal at the very start of an
    expression and these fragments are routinely embedded inside larger ones.
    """
    return r"^" + _MD_LEAD + pattern


def _line_decoration_end(raw: str, start: int) -> int:
    """Where the markdown decoration opening a line stops.

    Blockquote marks, heading hashes, bullet and numbered-list markers and a
    leading table pipe are structure rather than content. A newline is never
    consumed: a decorated empty line has to stay a line, or every anchor below
    it lands on the wrong row.

    Only the pipe that OPENS a row is consumed here, because this function
    describes a line's opening decoration. The cell separators inside the row
    are handled in _demarkdown, which is reached once per character; consuming
    only the leading one was how a pipe table's interior survived into the
    readable copy and defeated every key/value anchor at once.
    """
    i, n = start, len(raw)
    while i < n:
        j = i
        while j < n and raw[j] in " \t":
            j += 1
        if j >= n or raw[j] == "\n":
            break
        if raw[j] in ">#":
            while j < n and raw[j] in ">#":
                j += 1
        elif raw[j] in "-+*•":
            # A run rather than one character, so the '**' opening a bold bullet
            # and a '---' rule are both consumed here.
            while j < n and raw[j] in "-+*•":
                j += 1
        elif raw[j] == "|":
            j += 1
        elif raw[j].isdigit():
            k = j
            while k < n and raw[k].isdigit():
                k += 1
            if k < n and raw[k] in ".)":
                j = k + 1
            else:
                break
        else:
            break
        if j == i:
            break
        i = j
    while i < n and raw[i] in " \t":
        i += 1
    return i


def _demarkdown(raw: str) -> tuple[str, list[int]]:
    """The text with its decoration removed, and the map back to the original.

    The map holds one entry per character kept plus a sentinel, so a span found
    in the readable copy can always be turned back into a span of the original.
    """
    out: list[str] = []
    index: list[int] = []
    i, n = 0, len(raw)
    at_line_start = True
    while i < n:
        if at_line_start:
            end = _line_decoration_end(raw, i)
            at_line_start = False
            if end > i:
                i = end
                continue
        ch = raw[i]
        if ch == "\n":
            out.append(ch)
            index.append(i)
            i += 1
            at_line_start = True
            continue
        if ch == "|":
            # A table's cell separators are structure rather than content, so a
            # label and its value written into two cells read as one ordinary
            # line and a trailing pipe cannot defeat an end-anchored heading.
            # A space rather than nothing, so two cells cannot be glued into one
            # word, and one character out for one character in, so the map back
            # to the original stays exact.
            out.append(" ")
            index.append(i)
            i += 1
            continue
        if ch in _MD_MARKS:
            i += 1
            continue
        if ch == "_":
            j = i
            while j < n and raw[j] == "_":
                j += 1
            if i > 0 and raw[i - 1].isalnum() and j < n and raw[j].isalnum():
                for k in range(i, j):
                    out.append("_")
                    index.append(k)
            i = j
            continue
        out.append(ch)
        index.append(i)
        i += 1
    index.append(n)
    return "".join(out), index


class _Markup:
    """Model output paired with a decoration-free copy of itself.

    `text` is what every anchor is matched against; `raw` is what the reader is
    given. Bodies are sliced out of `raw` rather than out of `text`, because a
    citation stripped of its punctuation no longer names the document it cites.
    """

    __slots__ = ("raw", "text", "_index")

    def __init__(self, raw: str) -> None:
        self.raw = raw or ""
        self.text, self._index = _demarkdown(self.raw)

    def origin(self, pos: int) -> int:
        """The position in `raw` that `text[pos]` was taken from."""
        if pos < 0:
            return 0
        if pos >= len(self._index):
            return len(self.raw)
        return self._index[pos]

    def raw_line_start(self, pos: int) -> int:
        """Where the raw line holding `text[pos]` begins.

        A whole-line rewrite has to take the decoration that opens the line with
        it. The '**' of '**Disposition** - Substantiated' is consumed as
        decoration, so replacing from the mapped position alone left it stranded
        in front of the replacement and printed '****Disposition:**'.
        """
        return self.raw.rfind("\n", 0, self.origin(pos)) + 1

    def raw_block(self, start: int, end: int) -> str:
        """`raw` from `start` up to the beginning of the line `end` sits on.

        Used where `end` is the heading that closes the block: the decoration
        opening that heading's line belongs to the heading, not to the block
        above it.
        """
        if end >= len(self.text):
            return self.raw[self.origin(start):]
        return self.raw[self.origin(start):self.raw_line_start(end)]

    def raw_span(self, start: int, end: int) -> tuple[int, int]:
        """The tightest span of `raw` covering `text[start:end]`.

        Tight rather than loose, so that rewriting a value written '**Yes**'
        replaces the word and leaves the emphasis around it balanced.
        """
        if end <= start:
            here = self.origin(start)
            return here, here
        return self.origin(start), self.origin(end - 1) + 1


# A date span in the allegation used to add a sentence to the decomposition
# prompt asking whether the wording alleged more than one occasion. It is gone.
# Its trigger was a phrasing habit rather than a property of the claim, so it
# fired on some allegations and not on others in the same objective, and the
# only thing it could add was one more element to fail - a nudge that can move a
# disposition one way and never the other. ELEMENTS_TEMPLATE now puts the
# repetition rule to every allegation instead, in the same words each time.

# Anything in square brackets is read as a citation. A citation is circular when
# it names the claim under test instead of a document. The vocabulary is matched
# anywhere inside the bracket rather than only at its start, because "[see
# Allegation 2]" is the same circularity as "[Allegation 2]"; what keeps a
# document legitimately titled for the complaint out of the ban is the manifest
# test in _is_circular, not the position of the word or the presence of a page.
_CITE = re.compile(r"\[[^\]\n]{1,160}\]")
# The same, plus the parenthesised form. Two patterns rather than one because
# they are used for two different things: _CITE is what the scrubber EDITS, and
# deleting a parenthesis out of a sentence rewrites the sentence, while
# _CITE_ANY is only ever READ - to ask whether an element rests on a cited page
# and to count how much of a section does. A model that writes every citation
# in parentheses has departed from the template, but that is a habit of
# punctuation, and a habit of punctuation must not be able to decide a
# disposition.
_CITE_ANY = re.compile(r"\[[^\]\n]{1,160}\]|\([^)\n]{1,160}\)")
_CIRCULAR_INNER = re.compile(
    r"\b(?:allegation|complaint|objective|claim\s+under\s+test|this\s+task)\b",
    re.I)
# A page reference however it is spelled. Reading only "p.N" let a citation's
# spelling decide an element: "[Approvals_Log.pdf, page 7]" carries a page as
# plainly as "[Approvals_Log.pdf p.7]" does, and an element cited the first way
# was treated as citing nothing.
_PAGE_WORD = r"\b(?:pp?\.?|pages?|pg\.?)[ \t]*"
_PAGE_CITE = re.compile(_PAGE_WORD + r"\d", re.I)
# The same page, written with a separator instead of the word. A model that
# settles on "[Approvals_Log.pdf:7]" - an ordinary citation convention - was
# writing a citation this module read as carrying no page at all, and a block
# whose citations carry no page has its verdict withdrawn. These forms are
# accepted only inside a bracket that names an ingested document (see
# _carries_page), because on their own a colon and a number are as likely to be
# a time or a clause number as a page. A digit before the colon is refused for
# that reason: "10:30" is a time, not page thirty.
_PAGE_MARK = r"(?:(?<!\d)[:#][ \t]*|\bat[ \t]+)"
_PAGE_BARE = re.compile(_PAGE_MARK + r"\d{1,5}(?![:\d])", re.I)
# The page numbers themselves, for testing a citation against the page a fact
# was extracted from. A range cites every page it spans. Only ever applied to a
# bracket that has already been found to name an ingested document, which is
# what makes the bare forms safe to include here.
_PAGE_NUMBERS = re.compile(
    r"(?:" + _PAGE_WORD + r"(\d{1,5})|" + _PAGE_MARK + r"(\d{1,5})(?![:\d]))"
    r"(?:[ \t]*[-–—][ \t]*(\d{1,5}))?", re.I)

# Item markers. Numbering is tried first and bullets only when nothing was
# numbered, so a model that bullets the sub-lines of a numbered item does not
# have that item split into pieces.
#
# These two are the exception to the rule above: they are matched against the
# original text rather than against _Markup.text, because they split a section
# into the pieces that are then edited and reassembled, and splitting on the
# decoration-free copy would rebuild the section without its decoration. They
# carry their own tolerance for that reason alone.
_NUMBERED_ITEM = re.compile(r"^[ \t]*" + _MD_DECOR + r"\d+[.)]\*{0,2}[ \t]+", re.M)
_BULLET_ITEM = re.compile(r"^[ \t]*[-*+•]\*{0,2}[ \t]+", re.M)

SYSTEM_TEMPLATE = (
    "You are an investigating officer writing findings from a set of documents "
    "that have been collected and indexed. You write plainly and you do not "
    "overstate.\n"
    "\n"
    "Rules you follow without exception:\n"
    "- Every factual statement cites its source inline as [filename p.N].\n"
    "- You use only the material provided. You never rely on outside knowledge "
    "and never assume facts that the documents do not state.\n"
    "- Where the documents conflict, you say so explicitly, give both accounts "
    "with their sources, and do not silently prefer one.\n"
    "- An allegation is {yes_label} or {no_label} - those two labels and no "
    "others. The burden sits on the allegation: where the evidence does not "
    "carry every element of it, the disposition is {no_label} and what is "
    "missing is stated as a gap. A third label is a verdict on the "
    "investigation rather than on the claim, and you never write one.\n"
    "- The allegation and the complaint are the claim under test, never "
    "evidence. You never cite [Allegation N], the complaint, or this task in "
    "support of a finding. A finding with nothing else behind it is unsupported "
    "and is written as unsupported.\n"
    "- You distinguish what a document records first-hand from what it reports "
    "someone else said. Second-hand accounts are identified as such.\n"
    "- When the same fact exists in a primary record (a log, a form, a system "
    "report) and in a person's restatement of it, cite the record and its "
    "custodian first; the restatement is corroboration, not the source. Facts "
    "marked (RECORD OF EVIDENCE) come from the record itself - cite those "
    "before any interview that repeats the same figure.\n"
    "- When a witness states a limitation on their own observation - distance, "
    "an obstructed view, headphones, arriving mid-event - carry that limitation "
    "into any finding that rests on their account.\n"
    "- Two accounts conflict only when they describe the same event and take "
    "positions that cannot both be true. Accounts that agree corroborate each "
    "other and are reported as corroboration, never as conflict.\n"
    "- First and second person belong to the speaker of the document they "
    "appear in. You resolve them to a name before comparing two accounts.\n"
    "- An element that asserts repetition - a pattern, a course of conduct, "
    "repeated acts - is not carried by a single conceded instance.\n"
    "- You record what the documents establish whether or not it helps the "
    "allegation. A conceded act is a finding of fact even where the disposition "
    "is {no_label}.\n"
    "- Allegations are substantiated or not; an objective or goal is answered, "
    "never 'substantiated'. The numbered allegations you are given are the "
    "complete list - do not invent, split, or renumber them.\n"
    "- You do not recommend discipline, and you do not invent a motive for "
    "anyone. An interest in the outcome that a document actually records - a "
    "pending grievance, a dispute predating the complaint, discipline already "
    "under way, a benefit a witness stands to gain - is a fact of the record "
    "and you write it down wherever it bears on the weight of an account. "
    "Setting down what a document states is not speculation about motive; "
    "attributing a motive no document states is, and that you never do."
)

ELEMENTS_TEMPLATE = """Break this allegation into the separate propositions that
would each have to be true for it to be substantiated.

ALLEGATION: {allegation}

You are given the allegation and nothing else, deliberately. The elements come
from what is alleged, not from what the evidence happens to show; an element
that only exists because a document admits it is not an element of the claim.

An element is one proposition that could separately be true or false - who did
it, what act, to or with whom, in what capacity or position, under what
qualifier (on duty, without consent, using government resources, contrary to
policy), and how often. Take each element from the allegation's own words. Do
not add an element the allegation does not assert. Do not merge two separate
assertions into a single element.

A rule the allegation cites - an article, a regulation, an instruction, a
policy, a statute, or any acronym or short form standing for one - is the
authority the conduct is said to offend, not a proposition to be proved from
the documents. It is never an element, and a bare citation or acronym is never
an element on its own. An
investigation establishes what a person did; the corpus is a set of statements
and records about their conduct and will not contain the text of the rule, so
an element made out of a citation can only ever fail for want of evidence that
was never going to be there. Decompose the conduct the allegation describes and
leave the authority out of the table.

If the allegation's wording asserts that the conduct happened more than once - a
plural act, a frequency word, a pattern, a course of conduct - then one element
is the repetition itself, marked MULTIPLICITY: yes. Every other element is
marked MULTIPLICITY: no.

Output the elements and nothing else, in exactly this shape:

E1 | ELEMENT: <the proposition, in the allegation's own terms>
     WORDS: "<the exact words of the allegation this element comes from>"
     MULTIPLICITY: <yes|no>
"""

CONFLICT_TEMPLATE = """You are comparing witness accounts against each other. Below
are extracted facts and passages bearing on one allegation.

THE ALLEGATION THESE ACCOUNTS BEAR ON: {allegation}

You are not deciding it and you are not looking for support for it. It is here
so you can tell a disagreement that decides something from one that decides
nothing. Where accounts differ on the quality this allegation turns on -
whether an act was ordered or offered, done on duty or off, refused freely or
impossible to refuse, paid or unpaid - that difference is the one to report,
ahead of a difference about where a remark was made or how many people were
present. Report both kinds if you find both, decisive ones first.

Resolve the pronouns before you compare anything. "I", "me", "my" and "we"
belong to the speaker of the document they appear in; "you" belongs to the
person that speaker is addressing. The speakers are listed below. Rewrite each
account in the third person, with names in place of pronouns, and only then
decide whether two accounts disagree. If a document's speaker is not identified
below, do not report any conflict that rests on a pronoun in it.

A conflict requires BOTH of the following, and your entry must state both:
  (a) the same event, act, occasion or figure, described by both accounts; and
  (b) positions that cannot both be true at the same time.

These are NOT conflicts, and reporting one as a conflict is an error:
  - two accounts that agree with each other;
  - one account mentioning something the other does not - silence is not denial;
  - one account naming fewer people than another, unless it explicitly excludes
    the others ("only", "no one else", "nobody but") AND the speaker's own name
    has already been put in place of their first-person pronouns, so that the
    shorter list is genuinely shorter;
  - two accounts differing in a detail that neither of them denies.
All of those belong under CORROBORATION.

One account saying a person had no real choice and another saying the same act
was optional is a CONFLICT, not corroboration, even though both agree the act
happened. So is one account calling a thing an instruction and another calling
it an invitation. Filing those under CORROBORATION because the act is not in
dispute loses the only thing that was.

But a disagreement about the CHARACTER of an act is a conflict, not a detail.
Whether something was ordered or requested, compelled or voluntary, on duty or
off duty, refused without consequence or impossible to refuse, paid or
unpaid - two accounts can agree entirely on what physically happened and still
be irreconcilable about what it was. That disagreement is usually the whole
question an allegation turns on, so report it: A is one account of the act's
character, B is the other. Neither side denying the act happened is exactly why
this looks like a detail, and it is not one.

These three you report as conflicts even though one side is a record rather than
a second witness:
  - observation-limit: a witness stating a limit on their own observation -
    distance, an obstructed view, earbuds, arriving mid-event. Report these even
    when nobody contradicts them: they decide how much weight the account can
    carry. B is the part of the account the limit undercuts.
  - defence-contradicted: an explanation offered by the person under
    investigation that the records contradict. A is the defence, B is the
    evidence against it.
  - self-contradiction: an account contradicted by that same person's own words
    elsewhere. This includes a prior statement of theirs quoted back to them
    inside another document - an interviewer reading out an earlier written
    account, a record repeating what someone reported at the time. The two
    halves need not sit in two different documents; what makes it a conflict is
    that the same person said both things and both cannot be true. A person
    departing from their own earlier account is one of the most consequential
    things an investigation can record, so do not pass over it because both
    quotes came off the same page.

Number the items under each heading and use exactly this shape. Write NONE under
a heading that has no items.

CONFLICTS:
1. TYPE: <contradiction | wording-variance | observation-limit |
     defence-contradicted | self-contradiction>
   EVENT: <the one event, act or figure both sides are about, with its date if
     the documents state one>
   A: "<quote>" [filename p.N]
   B: "<quote>" [filename p.N]
   INCOMPATIBLE: <why A and B cannot both be true>

A and B are each a verbatim quotation with its own citation, and they are the
two things that cannot both be true. Writing a description of the disagreement
in B - "contradicts his earlier account", "this is inconsistent with the
record" - leaves the entry with only one position in it, and a reader cannot
adjudicate a conflict whose second side was never quoted. Say why they are
incompatible in the INCOMPATIBLE line, which is what that line is for.

CORROBORATION:
1. FACT: <the fact both accounts independently support>
   A: "<quote>" [filename p.N]
   B: "<quote>" [filename p.N]

Work through it systematically rather than reporting the first disagreement you
notice. Take each account of each event in turn and compare it against every
other account of that same event, including the same person's account given at
a different time. Most conflicts in an investigation are quiet: two accounts
that sound compatible read separately and cannot both be true read together.
Report every one you find, not one per event.

WHO IS SPEAKING IN EACH DOCUMENT:
{speakers}

FACTS:
{relationships}

PASSAGES:
{passages}
"""

ALLEGATION_TEMPLATE = """Write the findings for this one allegation only.

ALLEGATION {number} - THIS IS THE CLAIM UNDER TEST. It is not evidence, it
proves nothing, and it is never cited in support of a finding:
{allegation}

ELEMENTS THAT MUST EACH BE PROVED:
{elements}

Use this structure exactly. The word "Disposition" appears once, in the line
shown below, and nowhere else in your answer.

#### Allegation {number}: <short restatement>

**Elements and weighing**

<One block per element listed above, in the same order and keeping its number,
in exactly this shape:

E<n> | ELEMENT: <the element>
   SUPPORTING: <each item of evidence for this element with its [filename p.N],
     one per line; "none">
   OPPOSING: <each item of evidence against it with its [filename p.N], one per
     line, including anything showing the act was voluntary, permitted,
     compensated, declined without consequence, off duty, outside the capacity
     alleged, or otherwise not the conduct this element describes; any interest
     a source supporting it has in the outcome, such as discipline, a grievance
     or a dispute preceding the complaint; any retraction or correction a
     source made under questioning; and where a supporting account is
     secondhand - the speaker heard it from someone else, was not present, or
     states a limit on what they could observe - say so and say what it costs
     that account, because an account of what somebody else said is weaker
     evidence of the act than an account of the act; "none">
   INSTANCES ESTABLISHED: <a whole number - how many separate occasions the
     evidence actually establishes for this element>
   MET: <Yes|No> - <one sentence: weighing the two lists against each other, is
     this element more likely true than not>

Where any account bearing on this allegation is secondhand - a speaker
repeating what another person told them, or describing something they were not
present for - the section must say so and say what weight it was given, even
where the finding does not rest on it. An investigation records what it set
aside as much as what it relied on: a reader who cannot see that the secondhand
account was considered and discounted cannot tell it from one that was
overlooked.

Where a finding rests on an account of what somebody else said, rather than on
what the speaker saw or did or on a record, say so in the finding itself. An
investigation that cannot tell the two apart cannot weigh them, and a chain of
report and repetition can look like corroboration while adding nothing: the
same claim counted twice because two people repeated it.

Every finding here is about the conduct THIS allegation alleges. Where another
numbered allegation covers a different act - a statement made, a card used, an
order given - that act is answered in its own section and is not restated as a
finding here, however plainly the documents establish it. Repeating it puts the
same conduct under two allegations and makes the report read as though more was
found than was.

Weigh the opposing evidence, do not merely list it. An element asserting
repeated conduct is not met by one conceded instance.

Judge each element exactly as it is worded. Where the evidence establishes
something weaker than the element asserts - that people helped rather than that
they were directed to, that a thing was requested rather than ordered, that a
person was present rather than responsible - the element is NOT met, however
well that weaker fact is evidenced. Quietly substituting the weaker proposition
and meeting that instead is the commonest way an allegation is wrongly
substantiated, and it is invisible afterwards because the sentence reads as
though the element were met.

Where the opposing list is longer or carries more weight than the supporting
list, MET is No, unless the same sentence says why that opposing evidence does
not bear on the element as worded. "More likely true than not" is a comparison
between the two lists, not a judgement that the supporting list exists.>

**Disposition:** <{yes_label} | {no_label}>

<{yes_label} only if every element above is MET: Yes. If any element is MET: No
then the disposition is {no_label}. There is no third label: what the evidence
does not establish goes under Gaps, not into the disposition.>

**Findings**

<Numbered findings. One fact each, each ending with its [filename p.N]
citation. Record what the documents establish whether or not it helps the
allegation - an act the subject conceded is a finding of fact even where the
disposition is {no_label}. Never cite the allegation, the complaint or
this task as a source; a finding with nothing else behind it begins with
"UNSUPPORTED:".>

**Conflicts in the evidence**

<Adjudicate EVERY candidate listed under CANDIDATE CONFLICTS below - one
numbered entry each: confirm it, resolve it with a source, or explain why it is
not a real conflict. Keep a candidate as a conflict unless the two accounts
plainly agree: a comparison pass that had nothing else to do found it, and
dropping it silently costs a reader something they cannot recover, while
carrying one that turns out weak costs them a line they can judge for
themselves. Two accounts of the same event that agree, or that differ only in
what one of them does not mention, are not conflicts; say so and record them
under Corroboration below instead, never by deleting them. Include witness observation limits and any
defence the records contradict; both belong here rather than under Gaps. Add any
further conflicts you see. Write "None identified." only if the candidate list
was NONE and you find none yourself.>

**Gaps**

<What the documents do not establish, and the SPECIFIC record that would settle
it. Never call something a gap that a cited finding above already resolves, and
never name a record that already appears in the corpus listed below - those
documents are in evidence and their contents are not missing. Write
"None identified." if there are none.>

**Corroboration**

<Accounts that independently support the same fact - one numbered entry each,
naming the fact and both sources with their citations. Two witnesses agreeing is
weight behind a finding, not a conflict. This section comes last because it is
supporting weight rather than a disagreement to adjudicate, and because keeping
it outside the conflicts section is what stops an agreement from being read back
as one. Write "None identified." if there are none.>

---

CANDIDATE CONFLICTS (from a dedicated cross-witness comparison - adjudicate each):
{conflicts}

CANDIDATE CORROBORATION (accounts that agree - weight, not conflict):
{corroboration}

WHO IS SPEAKING IN EACH DOCUMENT:
{speakers}

DOCUMENTS IN THIS INVESTIGATION (the complete corpus - a record named here is
already in evidence and is not a gap):
{manifest}

RELATIONSHIPS EXTRACTED FROM THE DOCUMENTS:
{relationships}

PASSAGES FROM THE DOCUMENTS:
{passages}
"""

RETRY_NOTE_COVERAGE = """
The previous attempt at this section could not be read: its element blocks did
not cover the elements listed above, one block to each, so nothing in it settled
the disposition. Write the section again and keep the element blocks in exactly
the shape given - one block per element listed, its own number, its own MET
line - whatever else you write around them.
"""

# The blocks were there and the verdict line in them was not readable, which is
# a different defect and needs a different instruction. Told to fix its block
# coverage - which had not been wrong - the model reproduced the same section on
# the second draw and the allegation banked the procedural default. Both labels
# are named here, in the shape the template asks for, so the retry cannot be
# read as a hint about which one to write.
RETRY_NOTE_MET = """
The previous attempt at this section could not be read: its element blocks were
there, but the MET line inside them could not be read, so no element was weighed
and nothing in it settled the disposition. Write the section again and give
every element block a line of its own reading exactly "MET: Yes" or "MET: No",
followed by the one sentence weighing the two lists against each other.
"""

# The blocks and their verdicts were both readable and the verdict could not be
# accepted as written. What that asks for is the support, and it asks for it of
# a block weighed either way, because a verdict resting on nothing cannot be
# checked whichever way it reads.
RETRY_NOTE_SUPPORT = """
The previous attempt at this section could not be accepted: an element block
weighed its element without support this report can check - no citation at all,
or a citation naming a document that is not in the corpus listed above. Write
the section again and give every element block, whichever way it is weighed, at
least one citation in the form [filename p.N] naming a document from that
corpus. Where an element asserts that the conduct happened more than once, write
the number of occasions on its INSTANCES ESTABLISHED line.
"""

SUMMARY_TEMPLATE = """Write the opening of an investigation report.

Use this structure exactly:

## Summary of findings

<One short paragraph per allegation stating the disposition and the single most
important reason, each with its [filename p.N] citation. Use only the two labels
{yes_label} and {no_label}, exactly as they appear in the dispositions below: do
not restate them, soften them, or decide them again. Every citation
you write must appear verbatim in that allegation's FINDINGS AS WRITTEN below -
you have not been shown anything else, so a citation not in that block is one
you made up. Where an allegation's findings carry no citation at all, write that
the disposition rests on no cited finding and give no citation. Repeat that
allegation's BASIS line in your paragraph, in your own sentence, so a reader
learns how much of the element table was weighed and how much was cited before
they read the reason. Do not cite the allegation or the complaint as a source -
they are the claim under test. If the corpus note below is not "(none)", the
first sentence of this section states that the record is incomplete and names
the documents the note names.>

## Persons named

<A list. For each: name, role if the documents state one, and the documents
they appear in. Note where the same person appears under more than one form of
their name.>

## Timeline

<Dated events in order, each as: DATE — event [filename p.N]. Only dates the
documents actually state.>

---

CORPUS NOTE:
{corpus_note}

THE ALLEGATIONS AND THEIR DISPOSITIONS:
{dispositions}

DATED EVENTS EXTRACTED FROM THE DOCUMENTS:
{timeline}

PERSONS AND ORGANISATIONS EXTRACTED:
{entities}
"""


def disposition_labels() -> tuple[str, str]:
    """The two labels of record for this investigation, the affirmative first.

    The doctrine is fixed - two labels, the burden on the allegation, no third
    label for a record that is merely inconvenient - but the wording of the two
    is the appointing authority's, not this module's. It was hardcoded here and
    separately hardcoded in the scoring key, so app/ could only ever express the
    scheme it happened to be written beside; an investigation whose labels of
    record are "Sustained" and "Not sustained" had every disposition rendered in
    words its own authority does not use.

    Anything other than a pair of non-empty labels is ignored with a warning
    rather than half-applied: a partially readable setting would rename one
    label and leave the other, which reads on the page as a third label.
    """
    raw = state.get_setting(LABELS_KEY, "").strip()
    if not raw:
        return DISPOSITIONS
    try:
        items = [str(x).strip() for x in json.loads(raw)]
    except (json.JSONDecodeError, TypeError, ValueError):
        items = []
    if len(items) == 2 and all(items):
        return (items[0], items[1])
    log.warning("%s is set but is not a pair of non-empty labels; the built-in "
                "pair %s is used", LABELS_KEY, " / ".join(DISPOSITIONS))
    return DISPOSITIONS


def get_goal() -> str:
    return state.get_setting(GOAL_KEY, "")


def get_allegations() -> list[str]:
    raw = state.get_setting(ALLEGATIONS_KEY, "")
    if raw:
        try:
            items = json.loads(raw)
            return [str(a).strip() for a in items if str(a).strip()]
        except json.JSONDecodeError:
            pass
    # One-time migration from the legacy free-text block, if one exists.
    legacy = state.get_setting(OBJECTIVE_KEY, "")
    return split_allegations(legacy) if legacy else []


def set_objective(goal: str, allegations: list[str]) -> None:
    """The goal and each allegation are separate fields on the page now, so
    nothing is ever parsed out of prose - a misparse here once relabeled a
    substantiated allegation as 'insufficient' in the delivered summary."""
    state.set_setting(GOAL_KEY, (goal or "").strip())
    cleaned = [str(a).strip() for a in allegations if str(a).strip()]
    state.set_setting(ALLEGATIONS_KEY, json.dumps(cleaned, ensure_ascii=False))


def split_allegations(objective: str) -> list[str]:
    """One entry per allegation.

    Accepts numbered lists, bulleted lists, or one per line - operators write
    these by hand and should not have to match a format.
    """
    text = (objective or "").strip()
    if not text:
        return []
    # Numbered items are the allegations; anything before the first number is
    # the objective's preamble - context, never Allegation 1. Treating the
    # preamble as an allegation shifts every number down and misfiles findings
    # under the wrong heading, which a graded run demonstrated in practice.
    marker = re.compile(r"(?m)^\s*(?:allegation\s*)?(?:\d+[.)]|[-*•])\s+", re.IGNORECASE)
    matches = list(marker.finditer(text))
    if matches:
        items = []
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end():end].strip()
            if body:
                items.append(body)
        if items:
            return items
    return [line.strip() for line in text.splitlines() if line.strip()] or [text]


def _format_relationships(facts: list[dict]) -> str:
    if not facts:
        return "(none found for this allegation)"
    # Records first, then statements, then everything else - so the primary
    # source is what the model reads at the top of the list, not merely what it
    # was told to prefer. A person restating a log is corroboration; the log is
    # the evidence.
    KIND = {"record": 0, "appointment": 1, "statement": 2, "interview": 3,
            "notes": 4, "unknown": 5}
    # Within interviews the speaker's relationship to the evidence decides
    # precedence: the custodian of a system is the source for what it recorded,
    # and the subject repeating that figure is the furthest from it.
    ROLE = {"custodian": 0, "complainant": 1, "supervisor": 1, "witness": 2,
            "": 2, "subject": 3}
    ordered = sorted(facts, key=lambda f: (KIND.get(f.get("source_kind") or "unknown", 5),
                                           ROLE.get(f.get("source_role") or "", 2)))

    # Grouped by the day the event happened, precedence preserved inside each
    # day. A comparison has to put accounts of ONE event beside each other, and
    # a flat list ordered only by source precedence scatters the three accounts
    # of a single afternoon among forty other lines. Grouping does not change
    # what the model is given, only whether the things it must compare are
    # adjacent - which is the difference between reading a list and having to
    # hold it all in mind at once.
    shown = ordered[:MAX_RELATIONSHIPS]
    days: dict[str, list[dict]] = {}
    for f in shown:
        days.setdefault(str(f.get("event_date") or "")[:10], []).append(f)

    lines = []
    for day in sorted(days, key=lambda d: (d == "", d)):
        if len(days) > 1:
            lines.append(f"\n  {day or 'no date given'}:")
        for f in days[day]:
            when = f" on {f['event_date']}" if f.get("event_date") else ""
            kind = f.get("source_kind") or "unknown"
            role = f.get("source_role") or ""
            if kind == "record":
                tag = " (RECORD OF EVIDENCE)"
            elif role == "custodian":
                tag = " (CUSTODIAN OF THESE RECORDS)"
            elif role == "subject":
                tag = " (THE SUBJECT - restatement, not the record)"
            else:
                tag = ""
            lines.append(f"- {f['subject']} {f['predicate']} {f['object']}{when} "
                         f"[{f['source_file']} p.{f['source_page']}]{tag}")
    return "\n".join(lines)


def _format_passages(passages: list[dict], speakers: dict[str, str] | None = None) -> str:
    """Passages, each labelled with whose words they are.

    The speaker is carried on the passage header rather than left to the prose
    rule alone: a model cannot resolve "I" to a name it was never shown next to
    the text, and an unresolved first person is what turns one witness's account
    of an event into a contradiction of another's account of the same event.
    """
    if not passages:
        return "(no matching passages)"
    speakers = speakers or {}
    blocks = []
    for p in passages:
        name = speakers.get(p.get("doc_id"))
        who = f" (speaking: {name})" if name else ""
        blocks.append(f"[{p['filename']} p.{p['page_num']}]{who}\n{p['text'].strip()}")
    return "\n\n".join(blocks)


def _timeline_block() -> str:
    rows = graph.timeline() if graph.available() else []
    if not rows:
        return "(no dated events extracted)"
    return "\n".join(
        f"- {r['date']}: {r['subject']} {r['predicate']} {r['object']} "
        f"[{r['source_file']} p.{r['source_page']}]" for r in rows[:60])


def _entities_block() -> str:
    rows = state.query(
        """SELECT e.entity_id, e.entity_type, e.canonical_name, e.mention_count,
                  (SELECT GROUP_CONCAT(DISTINCT d.filename)
                     FROM triples t JOIN documents d ON d.doc_id = t.doc_id
                    WHERE t.subject_name = e.canonical_name
                       OR t.object_name = e.canonical_name) AS files
           FROM entities e
           WHERE e.merged_into IS NULL AND e.entity_type IN ('PERSON','ORG')
           ORDER BY e.mention_count DESC LIMIT 40""")
    if not rows:
        return "(no entities extracted)"
    out = []
    for r in rows:
        # merged_into holds the surviving row's entity_id, which is built by
        # entities.entity_id() from the NORMALISED name - lower case, no
        # punctuation, no rank. Rebuilding that key here as type:canonical_name
        # matched only when the canonical name happened to already be in that
        # form, so every merge the pipeline had made was invisible in the report
        # and a reader could not tell that two names in the record had been
        # treated as one person. The row's own id is the key, so it is asked for.
        aliases = state.query(
            "SELECT canonical_name FROM entities WHERE merged_into=?",
            (r["entity_id"],))
        also = ""
        if aliases:
            also = " (also appears as " + ", ".join(a["canonical_name"] for a in aliases) + ")"
        out.append(f"- {r['entity_type']}: {r['canonical_name']}{also} — "
                   f"mentioned {r['mention_count']}x in {r['files'] or 'unknown'}")
    return "\n".join(out)


def _manifest_block(docs: list[dict]) -> str:
    """The corpus, named to the model, so a gap cannot ask for what is in it.

    The symptom this answers was a report calling for a certified transaction
    history that had been ingested: a gap is only a gap when the record is
    absent, and the model can only know that if it is shown what is present.
    """
    if not docs:
        return "(no documents ingested)"
    return "\n".join(
        f"- {d['filename']} — {d.get('doc_kind') or 'unknown'} document, "
        f"status {d.get('status') or 'unknown'}, "
        f"{d.get('assertions') or 0} assertion(s) extracted" for d in docs)


def _speakers_block(docs: list[dict], speakers: dict[str, str]) -> str:
    if not docs:
        return "(no documents ingested)"
    lines = []
    for d in docs:
        name = speakers.get(d["doc_id"])
        lines.append(f"- {d['filename']}: {name}" if name else
                     f"- {d['filename']}: speaker not identified — do not "
                     f"resolve pronouns in this document")
    return "\n".join(lines)


def _integrity_banner(blocking: list[dict], silent: list[dict], total: int,
                      corpus_changed: bool = False) -> str:
    """The corpus-completeness warning, or "" when the corpus is whole.

    It is a blockquote at the very top of the report rather than a note at the
    bottom because its whole purpose is to be impossible to read past: a reader
    who reaches a disposition before learning the record was incomplete has
    already been misled by it.
    """
    seen = {b["doc_id"] for b in blocking}
    rows = list(blocking) + [s for s in silent if s["doc_id"] not in seen]
    if not rows and not corpus_changed:
        return ""
    lines = ["> **CORPUS INTEGRITY WARNING — THE FINDINGS BELOW MAY BE UNSAFE.**", ">"]
    if corpus_changed:
        lines += ["> The set of ingested documents changed while this report was "
                  "being written. The evidence below was gathered from the corpus "
                  "as it stood when generation began, so a document that arrived "
                  "or finished processing during the run contributed nothing to "
                  "it and is not named in the manifest the analysis read. "
                  "Re-run the report.",
                  ">"]
    if not rows:
        # Nothing to list, so the blockquote ends on its last sentence rather
        # than on the empty "> " that separates it from a list.
        return "\n".join(lines[:-1]) + "\n\n"
    if blocking:
        lines += ["> Generation was overridden while the corpus was still "
                  "unsettled. Every disposition below rests on a partial record.",
                  ">"]
    lines += [f"> {len(rows)} of {total} ingested document(s) contributed no "
              f"evidence to this report:", ">"]
    lines += [f"> - **{r['filename']}** — {r['reason']}" for r in rows]
    lines += [">",
              "> A gap named below may describe evidence that is present in the "
              "corpus but absent from this analysis. Re-run the pipeline for the "
              "documents named above before relying on any disposition."]
    return "\n".join(lines) + "\n\n"


def _normal_words(text: str) -> str:
    """Lower-case words only, so a filename and a citation to it can be compared."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _document_names(docs: list[dict]) -> set[str]:
    """Every ingested filename, reduced to the form a citation can be tested against.

    The model writes "[Allegation Letter p.3]" for a file ingested as
    "allegation_letter.pdf", so both sides are stripped of punctuation and
    extension and compared as words. Names shorter than three characters are
    dropped: a one- or two-letter stem matches inside almost any bracket and
    would wave every citation through.
    """
    names = set()
    for doc in docs:
        stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", str(doc.get("filename") or ""))
        normal = _normal_words(stem)
        if len(normal) >= 3:
            names.add(normal)
    return names


def _token_run(tokens: list[str], wanted: list[str]) -> bool:
    """True when `wanted` appears as consecutive whole tokens of `tokens`."""
    if not wanted or len(wanted) > len(tokens):
        return False
    return any(tokens[i:i + len(wanted)] == wanted
               for i in range(len(tokens) - len(wanted) + 1))


def _names_document(citation: str, documents: set[str]) -> bool:
    """True when an ingested document's name appears inside the bracket.

    Whole tokens in sequence, not a substring. Substring containment matched a
    short document stem inside a longer word - "log" inside "dialogue" and
    "catalogued", "memo" inside "memorandum" - so a bracket that named no
    document at all was read as naming one. That let a citation of the
    allegation back at itself pass the circularity test and be counted among the
    grounded citations printed beside the disposition.
    """
    tokens = _normal_words(citation).split()
    return any(_token_run(tokens, name.split()) for name in documents)


def _carries_page(citation: str, documents: set[str]) -> bool:
    """True when the bracket names a page of something.

    One test for every caller, rather than one regex per caller: the tolerance
    a citation is read with decides whether an element was weighed at all, so
    two callers reading it differently is two different answers to the same
    question. The bare separator forms are admitted only where the bracket also
    names an ingested document, so a stray number in prose stays a stray number.
    """
    if _PAGE_CITE.search(citation):
        return True
    return bool(_PAGE_BARE.search(citation)) and _names_document(citation, documents)


def _cited_pages(text: str, documents: set[str]) -> set[tuple[str, int]]:
    """(document name, page) for every bracket naming an ingested document."""
    found: set[tuple[str, int]] = set()
    for citation in _CITE_ANY.findall(text):
        tokens = _normal_words(citation).split()
        named = [n for n in documents if _token_run(tokens, n.split())]
        if not named:
            continue
        for match in _PAGE_NUMBERS.finditer(citation):
            # Group 1 is the page written with the word, group 2 the page
            # written with a separator; exactly one of them is present.
            first = int(match.group(1) or match.group(2))
            last = int(match.group(3)) if match.group(3) else first
            # A descending or implausibly wide range is a typo rather than a
            # citation to two hundred pages, so it cites its first page only.
            if last < first or last - first > 40:
                last = first
            for page in range(first, last + 1):
                for name in named:
                    found.add((name, page))
    return found


def _is_circular(citation: str, documents: set[str]) -> bool:
    """A bracket naming the claim under test rather than a document.

    The claim's vocabulary anywhere inside the bracket raises the question and
    the corpus answers it: a bracket that names an ingested document is a
    citation to that document however that document happens to be titled, and a
    bracket that names none is the allegation cited in support of itself.

    Testing against the corpus replaces two heuristics that were each wrong in
    one direction. Requiring the vocabulary at the start let "[see Allegation 2]"
    through, and requiring the absence of a page number let "[Allegation 2 p.1]"
    buy its way out of the ban by adding one - while both together deleted
    "[Allegation Letter 3 Feb]", which in an investigation whose complaint letter
    is an exhibit is the citation the finding actually rests on.
    """
    inner = citation[1:-1]
    if not _CIRCULAR_INNER.search(inner):
        return False
    return not _names_document(inner, documents)


def _grounded_citations(text: str, documents: set[str]) -> int:
    """Citations carrying a page and naming a document that was ingested.

    Counted rather than corrected. It is the measure of how much of a section
    rests on the corpus at all, which is printed beside the disposition so that
    a verdict resting on nothing is visible without reading the section.
    """
    return sum(1 for c in _CITE_ANY.findall(text)
               if _names_document(c, documents) and _carries_page(c, documents))


def _flat_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


_FINDING_CITE = re.compile(r"\[([^\]\s,]+)[^\]]*?p{1,2}\.?\s*(\d+)[^\]]*\]",
                           re.I)
_FINDING_QUOTE = re.compile(r"[\"\u201c]([^\"\u201c\u201d]{12,}?)[\"\u201d]")


def _foreign_refs(line: str, tagged: list[dict]) -> set[int]:
    """Allegations the evidence behind this finding was filed under.

    A finding quotes its evidence sometimes and paraphrases it the rest of the
    time, so quotation alone finds only half of them. The paraphrase is matched
    on the distinctive words of the assertion, and the bar is deliberately high
    - most of them, not a few - because dropping a finding that belonged here
    is worse than leaving one that did not.
    """
    text = _flat_text(line)
    refs: set[int] = set()

    # The citation is the third signal and the only one that survives a
    # paraphrase. Where every tagged assertion on a cited page belongs to one
    # allegation, that page IS that allegation's answer - the stretch of
    # transcript following its marker - and a finding resting on it is resting
    # on another allegation's section however it is worded. Pages carrying no
    # markers, or markers for more than one allegation, say nothing here.
    for doc, page in _FINDING_CITE.findall(line):
        # The citation carries the filename with its extension; the document
        # id does not. Match on whichever is the prefix of the other so a
        # trailing ".pdf" cannot make a page look uncited.
        stem = _flat_text(doc).rsplit(".", 1)[0]
        on_page = [f for f in tagged
                   if stem and str(f.get("page_num")) == page
                   and (_flat_text(f.get("doc_id")).startswith(stem)
                        or stem.startswith(_flat_text(f.get("doc_id"))))]
        if len(on_page) < 2:
            continue
        page_refs = set()
        for f in on_page:
            page_refs |= {int(x) for x
                          in re.findall(r"\d+", str(f["allegation_ref"]))}
        if len(page_refs) == 1:
            refs |= page_refs

    quotes = [_flat_text(q) for q in _FINDING_QUOTE.findall(line)]
    for f in tagged:
        quote = _flat_text(f.get("quote") or "")
        if not quote:
            continue
        matched = any(q in quote or quote in q for q in quotes)
        if not matched:
            words = [w for w in quote.split() if len(w) > 4]
            if len(words) >= 5:
                hits = sum(w in text for w in words)
                matched = hits >= max(5, int(len(words) * 0.6))
        if matched:
            refs |= {int(x) for x in re.findall(r"\d+", str(f["allegation_ref"]))}
    return refs


CONFLICT_DRAWS = 5
# The comparison pass samples even when the rest of the report is decoded
# greedily. Its draws are pooled, and pooling only buys something when the
# draws differ: at temperature 0 five draws are five copies of one, and the
# pass finds whatever a single greedy comparison finds. Everywhere else the
# opposite is wanted - a disposition should not move because a token was
# sampled - so the temperature is raised here and nowhere else. Coverage is
# the goal in this pass; the precision check downstream is what makes an
# over-eager draw safe.
CONFLICT_TEMPERATURE = 0.5


def _merge_candidates(blocks: list[str]) -> str:
    """Pool numbered items from several passes, dropping repeats.

    One comparison pass finds some of the disagreements in a record and a
    second finds others: across runs of one unchanged corpus this pass has
    returned a different pair of conflicts each time, which says the limit is
    the sampling and not the evidence. Independent draws pooled together
    recover what any single draw misses, and the adjudication downstream throws
    out anything that does not hold - so a pass that is thorough and sometimes
    wrong is worth more here than one that is careful and incomplete.

    Two items are the same when they quote the same first position, which is
    what survives rewording between draws.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for block in blocks:
        if not block or block.strip().upper() == "NONE":
            continue
        for item in re.split(r"(?m)^(?=\s*\d+\.\s)", block):
            if not item.strip():
                continue
            quotes = _FINDING_QUOTE.findall(item)
            fingerprint = _flat_text(quotes[0])[:90] if quotes else _flat_text(item)[:90]
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            merged.append(item.strip())
    if not merged:
        return "NONE"
    return "\n".join(f"{n}. {re.sub(r'^\s*\d+\.\s*', '', item)}"
                     for n, item in enumerate(merged, 1))


_SECONDHAND = re.compile(
    r"\b(told me|told him|told her|told them|heard about|heard that|"
    r"i heard|we heard|said that .{0,40}\bsaid\b|passed on|relayed|"
    r"secondhand|second hand|not firsthand|nothing firsthand|"
    r"i was not there|was not present|did not see it)\b", re.I)


def _belongs_to_another(line: str, allegations: list[str], index: int) -> int:
    """The allegation this finding actually describes, when it is not this one.

    A finding can restate another allegation's conduct without quoting any
    assertion tagged to it - the document it cites may carry no allegation
    markers at all, which leaves the tag-based check nothing to work with. What
    is always available is the allegations' own words: a finding that reads as
    a restatement of allegation 3 belongs in allegation 3's section, and its
    appearance here counts the same conduct twice.

    Deliberately requires a clear margin. Allegations of one case share
    vocabulary - the same people, the same dates, the same card - so a narrow
    lead means the finding is simply about the case, and moving it on that
    basis would scatter findings between sections at random.
    """
    # The citation is not part of what the finding says, and a filename full
    # of case identifiers matches nothing usefully. Punctuation has to go too:
    # "2026," and "charges." are the same words as 2026 and charges, and
    # leaving the comma on made a date match a date and little else.
    bare = re.sub(r"\[[^\]]*\]", " ", line)
    words = [w for w in
             (t.strip(".,;:()\"'") for t in _flat_text(bare).split())
             if len(w) > 4]
    if len(words) < 5:
        return 0
    scores = []
    for n, text in enumerate(allegations, 1):
        terms = {w for w in
                 (t.strip(".,;:()\"'") for t in _flat_text(text).split())
                 if len(w) > 4}
        hit = sum(w in terms for w in words) if terms else 0
        scores.append((hit / len(words), hit, n))
    scores.sort(reverse=True)
    best, best_hits, best_n = scores[0]
    mine = next((sc for sc, _, n in scores if n == index), 0.0)
    # A clear margin, and standing on more than a single word. The floor is a
    # count rather than a share because an allegation is written in a line or
    # two: every share against so few words is small, and a share-based floor
    # rejected a finding that matched another allegation on twice as many
    # words as its own.
    if best_n != index and best_hits >= 2 and best >= mine * 1.5:
        return best_n
    return 0


def _secondhand_facts(number: int) -> list[dict]:
    """Secondhand assertions bearing on this allegation, from the whole corpus.

    Not from the retrieved evidence. Retrieval ranks a passage by how much it
    resembles the allegation, and "I heard from someone that finance flagged
    him" resembles almost nothing - so the hearsay a report most needs to say
    it discounted is exactly the material retrieval leaves behind. What was
    considered and set aside is a question about the corpus, so it is asked of
    the corpus.
    """
    try:
        rows = state.query(
            "SELECT subject_name, quote, doc_id, page_num, allegation_ref "
            "FROM triples")
    except Exception:
        return []
    out = []
    for r in rows:
        ref = str(r["allegation_ref"] or "")
        if ref and ref.strip() != str(number):
            continue
        quote = str(r["quote"] or "")
        if _SECONDHAND.search(quote):
            out.append({"subject": r["subject_name"], "quote": quote,
                        "source_file": r["doc_id"], "source_page": r["page_num"]})
    return out


def _secondhand_note(facts: list[dict]) -> str:
    """A line recording the secondhand material, or "" when there is none.

    An investigation records what it set aside as much as what it relied on.
    Where an account of this allegation is one person repeating another's
    words, a reader has to be told - otherwise a chain of report and
    repetition reads as corroboration, the same claim counted once for every
    person who passed it along. Written from the evidence rather than asked
    for in the prompt, because a disclosure that appears only when the model
    remembers it is not a disclosure a reader can rely on.
    """
    hits = []
    for f in facts:
        quote = str(f.get("quote") or "")
        if _SECONDHAND.search(quote):
            who = str(f.get("subject") or f.get("subject_name") or "").strip()
            cite = (f" [{f['source_file']} p.{f['source_page']}]"
                    if f.get("source_file") else "")
            hits.append(f"{who}: \"{quote[:110]}\"{cite}" if who else quote[:110])
    if not hits:
        return ""
    listed = "\n".join(f"  - {h}" for h in hits[:4])
    return ("\n\n*Secondhand accounts considered and given no independent "
            "weight — an account of what another person said is evidence that "
            "they said it, not that the thing happened:*\n" + listed + "\n")


def _drop_foreign_findings(section: str, facts: list[dict], number: int,
                           allegations: list[str] | None = None
                           ) -> tuple[str, list[str]]:
    """Remove findings that restate a different allegation's conduct.

    Each allegation is answered in its own section, and the act another
    allegation is about - a statement made, a card used, an order given - is
    established there. Repeating it here files one act under two allegations
    and makes the report read as though more was found than was, which is the
    error a reader is least able to catch, because every line of it is true.

    A finding is foreign when the evidence behind it was filed under other
    allegations and under none of this one. An untagged assertion is available
    to every allegation and never makes a finding foreign, so a document with
    no allegation markers cannot have its evidence withheld from anywhere.
    """
    tagged = [f for f in facts if str(f.get("allegation_ref") or "").strip()]
    if not tagged:
        return section, []
    kept, dropped = [], []
    for line in section.splitlines():
        refs = _foreign_refs(line, tagged) if line.strip() else set()
        elsewhere = (_belongs_to_another(line, allegations, number)
                     if allegations and line.strip() else 0)
        if elsewhere:
            dropped.append(line.strip()[:70])
            continue
        if refs and number not in refs:
            dropped.append(line.strip()[:70])
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def _scrub_allegation_citations(section: str,
                                documents: set[str]) -> tuple[str, int]:
    """Remove citations that point back at the allegation, and mark what is left.

    The allegation is the claim under test; a finding resting on it rests on
    nothing. Where a finding loses its only citation this way, it is relabelled
    unsupported rather than deleted - the model's claim is still on the page,
    correctly described.

    Numbered items first and bulleted ones when nothing was numbered, which is
    the order _split_items uses for the same reason. Split on numbering alone,
    the relabel never reached a bulleted findings list: the circular bracket was
    still deleted by the sweep at the foot of this function, so a claim resting
    on the allegation lost the one mark of what it rested on and reached the
    summary reading like a supported finding.
    """
    removed = 0
    parts = re.split("(" + _NUMBERED_ITEM.pattern + ")", section, flags=re.M)
    if len(parts) == 1:
        parts = re.split("(" + _BULLET_ITEM.pattern + ")", section, flags=re.M)
    out = [parts[0]]
    for i in range(1, len(parts), 2):
        marker = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        cites = _CITE.findall(body)
        circular = [c for c in cites if _is_circular(c, documents)]
        if circular:
            removed += len(circular)
            for c in circular:
                body = body.replace(c, "")
            survivors = [c for c in cites
                         if c not in circular and _carries_page(c, documents)]
            if not survivors:
                body = ("UNSUPPORTED — the allegation is the claim under test, "
                        "not evidence: " + body.lstrip())
            body = re.sub(r"[ \t]{2,}", " ", body)
        out += [marker, body]
    text = "".join(out)

    # Anything outside an item is scrubbed too; prose citing the allegation is
    # the same circularity without the marker in front of it.
    def strip(match: re.Match) -> str:
        nonlocal removed
        if _is_circular(match.group(0), documents):
            removed += 1
            return ""
        return match.group(0)

    return _CITE.sub(strip, text), removed


# The element identifier in its plain form. Everything the model puts around it
# - a leading table pipe, bold on the identifier alone, a bullet, a heading, a
# list number - has already been removed by _Markup before these are matched, so
# '| **E1** | ELEMENT: x |' and 'E1 | ELEMENT: x' reach them identically, and a
# table row's interior pipes have become spaces on the way here.
#
# What separates the identifier from the element it heads is a class of
# characters rather than one of them. Requiring the literal pipe meant 'E1:',
# 'E1.' and 'E1 -' - the same table restated the way a markdown-trained model
# restates it - parsed as no element at all, and an element nobody parsed is an
# element nobody weighed, which lands the whole allegation on the procedural
# default. 'Element 1' is accepted beside 'E1' because it is the same identifier
# spelled out.
#
# Whitespace separates them too, because the pipe of a real table row IS
# whitespace by the time these are matched - but a run of it, or a single space
# in front of the ELEMENT label the template asks for. One unremarkable space
# would have made a head out of any sentence opening with an element's name, and
# a sentence inside a block ("E2 was not met: no waiver is on file") would then
# have become a block of its own carrying that reading as its verdict.
_ELEMENT_SEP = (r"(?:[ \t]*[|:.)\-–—][ \t]*|[ \t]{2,}"
                r"|[ \t]+(?=ELEMENT[ \t]*:))")
_ELEMENT_HEAD = _anchored(r"(?:Element|E)[ \t]*(\d+)" + _ELEMENT_SEP)
_ELEMENT_NEXT = _anchored(r"(?:Element|E)[ \t]*\d+" + _ELEMENT_SEP)


def _parse_elements(text: str) -> list[dict]:
    """The decomposition pass's output, one dict per element.

    Read off the decoration-free copy, so a block written
    '**E1** | **ELEMENT:** x' with '**MULTIPLICITY:** **yes**' under it is the
    same block as the plain one. An unreadable MULTIPLICITY line is recorded as
    unread rather than quietly taken for "no": the repetition test hangs off
    that field, and a field the parser could not read is not an answer.
    """
    markup = _Markup(text or "")
    items: list[dict] = []
    for match in re.finditer(
            _ELEMENT_HEAD + r"[ \t]*ELEMENT[ \t]*:[ \t]*(.*?)"
            r"(?=" + _ELEMENT_NEXT + r"|\Z)", markup.text, re.I | re.M | re.S):
        body = match.group(2)
        head = re.split(_anchored(r"(?:WORDS|MULTIPLICITY)[ \t]*:"),
                        body, flags=re.I | re.M)[0]
        proposition = " ".join(head.split()).strip()
        if not proposition:
            continue
        words = re.search(r"WORDS[ \t]*:[ \t]*(.+)", body, re.I)
        mult = re.search(r"MULTIPLICITY[ \t]*:[ \t]*(yes|no)\b", body, re.I)
        if mult is None:
            log.warning("element E%d of the decomposition carries no readable "
                        "MULTIPLICITY line; it is recorded as unread",
                        len(items) + 1)
        items.append({
            "n": len(items) + 1,
            "text": proposition,
            "words": words.group(1).strip().strip('"').strip() if words else "",
            "multiplicity": bool(mult and mult.group(1).lower() == "yes"),
            "multiplicity_read": mult is not None})
    return items


def _elements_for(allegation: str, raw: str) -> tuple[list[dict], str]:
    """Elements for one allegation, and a note when the pass produced none.

    The decomposition pass is the only source of elements, and its own
    MULTIPLICITY answer is the only source for whether an element asserts
    repetition. It reads the allegation with no evidence in front of it, which
    is what stops it from reading the elements back off whatever the record
    happens to admit.

    An earlier version added a repetition element of its own whenever a closed
    word list matched the allegation's wording. That list matched quantifiers of
    objects ("multiple vehicles", "several thousand dollars") as readily as
    frequencies of occasions, and every one of its false positives appended an
    element the allegation did not assert and that a single-occasion record
    could not carry - so its errors all pushed one way, toward Not substantiated.
    A mechanism that can only move a disposition in one direction is a thumb on
    the scale rather than a check, so it is gone. Repetition is now whatever the
    decomposition marked as repetition, tested below against the occasions the
    documents actually carry.

    A pass that returns nothing usable degrades to the allegation itself as a
    single element rather than failing the report, and says so in the returned
    note: an undecomposed allegation is one MET: Yes away from Substantiated,
    and the reader has to be able to see that it was never split.
    """
    items = _parse_elements(raw)
    if items:
        unread = [item["n"] for item in items if not item["multiplicity_read"]]
        if not unread:
            return items, ""
        # Not a silent default. An element whose MULTIPLICITY line could not be
        # read is carried as asserting no repetition, which is the reading that
        # applies no occasions test to it, so the reader is told which elements
        # those were and the caller draws the decomposition again before
        # settling for it.
        log.warning("the decomposition carries no readable MULTIPLICITY line "
                    "for element(s) %s", ", ".join(f"E{n}" for n in unread))
        return items, (
            "*The decomposition pass gave no readable MULTIPLICITY line for "
            "element" + ("s " if len(unread) > 1 else " ") +
            ", ".join(f"E{n}" for n in unread) +
            ", so " + ("they were" if len(unread) > 1 else "it was") +
            " weighed as asserting no repetition and the occasions test was "
            "not applied to " + ("them" if len(unread) > 1 else "it") +
            ". This records what could not be read, not a finding that the "
            "conduct happened once.*")
    log.warning("element decomposition returned nothing usable; falling back "
                "to the allegation as a single element")
    return ([{"n": 1, "text": allegation.strip(), "words": allegation.strip(),
              "multiplicity": False, "multiplicity_read": False}],
            "*This allegation was not decomposed: the element pass returned "
            "nothing readable, so the allegation was weighed as a single "
            "proposition. Read the weighing below with that in mind - an "
            "allegation tested undecomposed can be carried by evidence that "
            "answers only part of it.*")


def _elements_block(items: list[dict]) -> str:
    lines = []
    for item in items:
        lines.append(f"E{item['n']} | ELEMENT: {item['text']}")
        if item["words"]:
            lines.append(f'     WORDS: "{item["words"]}"')
        lines.append(f"     MULTIPLICITY: {'yes' if item['multiplicity'] else 'no'}")
    return "\n".join(lines)


# Key/value anchors, written plainly because they are matched against
# _Markup.text. The shape that defeated the hand-tolerated versions was the
# commonest one a markdown-trained model writes - '- **MET:** **Yes**', with the
# colon inside the label's emphasis and the value emphasised separately - and
# every one of those blocks came back unweighed.
_MET_LINE = re.compile(_anchored(r"MET\b[ \t]*:[ \t]*(yes|no)\b"), re.I | re.M)
_MET_LOOSE = re.compile(r"\bMET[ \t]*:[ \t]*(yes|no)\b", re.I)
# The whole value, to the end of its line, rather than its first token. Reading
# one token turned "no fewer than three" into the count "no", which the table
# below mapped to zero: a phrase meaning at least three read as none, and a
# count of none forces the element to fail. Every misreading a one-token scan
# could produce was of that kind, so the parser could only ever fail an element
# on its phrasing.
_INSTANCES_LINE = re.compile(
    _anchored(r"INSTANCES[ \t]+ESTABLISHED[ \t]*:[ \t]*([^\n]*)"), re.I | re.M)
_INSTANCES_LOOSE = re.compile(
    r"INSTANCES[ \t]+ESTABLISHED[ \t]*:[ \t]*([^\n]*)", re.I)
# Counts written as words. A count the parser cannot read is treated below as no
# count at all, so failing to read "one" would leave an element unweighed on a
# spelling rather than on the evidence.
#
# "no" is not in this table. It is a negation particle rather than a numeral -
# it is the opening of "no fewer than", "no less than", "no more than" - and
# every one of those phrases carries its real numeral later in the same line.
# "none" and "zero" stay, because those two ARE the word for a count of nothing
# and a block writing one of them has stated a count.
_WORD_COUNTS = {"none": 0, "zero": 0, "one": 1, "once": 1, "single": 1,
                "two": 2, "twice": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
                "twelve": 12}
# A strict lower bound states its count from below: "more than one occasion" is
# two occasions or more, and reading it as one failed the element on the way the
# sentence was built rather than on what it said.
_MORE_THAN = re.compile(
    r"\b(?:more than|greater than|over|above|in excess of)\b", re.I)


# A table can carry the verdict in a cell under a "Met" column heading, with no
# label on the value of its own: '| E1 | ELEMENT: x | [memo.pdf p.2] | Yes |'
# normalises to a head line ending in a bare "Yes". That is an ordinary way to
# render the table the template asks for, so it is read - but only on the
# element's own head line, only when that line labels no verdict of its own, and
# only where the cell boundary left the run of whitespace that a separator
# becomes, so a sentence ending "yes or no" is not a verdict. "Yes" and "No" are
# equally readable there, so this can settle a block either way.
_TRAILING_VERDICT = re.compile(r"[ \t]{2,}(yes|no)[ \t]*\Z", re.I)


def _head_line_verdict(body: str) -> re.Match | None:
    """A bare Yes/No in the last cell of the element's own head line."""
    head = body.split("\n", 1)[0]
    if re.search(r"\bMET\b", head, re.I):
        return None
    return _TRAILING_VERDICT.search(head)


def _met_in(body: str) -> re.Match | None:
    """The element's verdict: the last MET line in its block.

    Last rather than first, and a line-anchored match in preference to a loose
    one, because the block's own prose contains the word. An OPPOSING line
    reading "the policy requirement was not met: no waiver is on file" precedes
    the real verdict and was being read as it.
    """
    found = list(_MET_LINE.finditer(body)) or list(_MET_LOOSE.finditer(body))
    return found[-1] if found else None


def _instances_in(body: str) -> int | None:
    """The count of occasions the block states, or None when it states none.

    Every numeral on the line is read, in digits or in words, and the largest is
    the count. A value carrying more than one numeral is a range or a rate -
    "no fewer than three", "one per week for four weeks", "two or three" - and
    in each of those the largest numeral is the number of occasions the sentence
    is asserting while the smaller one is the unit or the other side of the
    bound. A strict lower bound is raised by one, because "more than one" is two
    or more.

    Citations are removed from the value first: a page number written beside the
    count ("1 [memo p.4]") is not a second reading of the count, and letting it
    be one would have defeated the repetition test with an ordinary citation.

    Reading a count too HIGH leaves the block's own verdict standing, whichever
    way it was written; reading one too low overrides it and records the element
    as not met. Those are not the same kind of error, so where the reading is in
    doubt this takes the higher one and declines to override. A line carrying no
    numeral at all returns None and is treated exactly like an absent line -
    there is then nothing to do the arithmetic on, which the caller says out
    loud rather than resolving.
    """
    found = _INSTANCES_LINE.search(body) or _INSTANCES_LOOSE.search(body)
    if not found:
        return None
    value = _CITE_ANY.sub(" ", found.group(1))
    counts = [int(t) if t.isdigit() else _WORD_COUNTS[t.casefold()]
              for t in re.findall(r"[A-Za-z0-9]+", value)
              if t.isdigit() or t.casefold() in _WORD_COUNTS]
    if not counts:
        return None
    return max(counts) + 1 if _MORE_THAN.search(value) else max(counts)


# A heading line: the word, an optional colon, and at most a few more words of
# its own title ("Findings of fact", "Conflicts in the evidence"). Requiring the
# word to end the line meant 'Findings:' - ordinary markdown - was not a heading
# at all, and requiring nothing at all after it would let a SUPPORTING sentence
# opening with one of these words pass for a heading. A line carrying a citation
# is evidence rather than a heading, so the bracket is excluded here.
_HEADING_TAIL = r"(?:[ \t]+[A-Za-z]+){0,3}[ \t]*:?[ \t]*$"
_BREAK_WORDS = r"(?:Findings|Conflicts|Gaps|Corroboration)"
# Where the element table stops. Without it the last element's block ran to the
# end of the section and swallowed the Findings, Conflicts and Gaps that follow
# it, so the last element was weighed on their citations and could be read off a
# "met" in their prose.
#
# The heading has to look like one from outside as well as inside. The tail
# above admits three words of title, so an unlabelled body line inside a block -
# "Gaps in the log" written under SUPPORTING - closed the table it was sitting
# in and left every element below it unweighed. So a line carrying its own title
# words is a break only where a blank line separates it from what precedes it,
# the way a heading is separated from the section above it; the bare word on its
# own line stays a break wherever it appears, because nothing else is written
# that way. The disposition line is unchanged: it is matched on its own
# punctuation rather than on a tail, so a body line cannot imitate it.
_SECTION_BREAK = (
    r"(?:^" + _MD_LEAD + r"(?:Disposition[ \t]*[:\-–—]|" + _BREAK_WORDS +
    r"\b[ \t]*:?[ \t]*$)"
    r"|(?<=\n\n)" + _MD_LEAD + _BREAK_WORDS + r"\b" + _HEADING_TAIL + r")")


def _parse_element_rows(section: str) -> list[dict]:
    """The written section's element blocks, with the spans needed to correct one.

    Matched against the decoration-free copy of the section and reported in the
    coordinates of the delivered one, so a block written as a markdown table row
    is the same block as a plain one and a correction still lands in the text
    the reader sees.

    The scan stops at the first section heading that follows the table. It used
    to run to the end of the section, and _SECTION_BREAK only closed the body of
    the block it was in, so any later line beginning 'E<n> |' was read as
    another element row: a Corroboration entry recapping the elements, or a Gaps
    line quoting one, voted on the disposition after the table had closed, and
    the last such line won. Prose written below the disposition could therefore
    overturn the table the disposition was bound to.
    """
    markup = _Markup(section)
    text = markup.text
    first = re.search(_ELEMENT_HEAD, text, re.I | re.M)
    if first:
        # Only a break AFTER the table can end it. A model that writes its
        # disposition line above the elements would otherwise truncate the
        # section before its own table and leave every element unweighed.
        stop = re.compile(_SECTION_BREAK, re.I | re.M).search(text, first.end())
        if stop:
            text = text[:stop.start()]
    rows = []
    for match in re.finditer(
            _ELEMENT_HEAD + r"(.*?)(?=" + _ELEMENT_NEXT + r"|\Z)",
            text, re.I | re.M | re.S):
        body = match.group(2)
        met = _met_in(body) or _head_line_verdict(body)
        met_span = None
        if met:
            at = match.start(2)
            met_span = markup.raw_span(at + met.start(1), at + met.end(1))
        rows.append({
            "n": int(match.group(1)),
            "body": markup.raw_block(match.start(2), match.end(2)),
            "met": (met.group(1).lower() == "yes") if met else None,
            "instances": _instances_in(body),
            "met_span": met_span})
    return rows


_MONTH_NUMBER = {}
for _i, _name in enumerate(
        ("january february march april may june july august september october "
         "november december").split(), 1):
    _MONTH_NUMBER[_name] = _i
    _MONTH_NUMBER[_name[:3]] = _i
# The one month whose usual abbreviation is neither its name nor three letters.
_MONTH_NUMBER["sept"] = 9
_MONTH_WORD = "|".join(sorted(_MONTH_NUMBER, key=len, reverse=True))
# The three ways a date is written in these documents: the ISO form the
# extractor emits, a numeric form, and a month named in words with the day on
# either side of it or absent altogether. Month precision counts: a fact the
# extractor could only place in a month is still a fact about a different
# occasion from one in another month.
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})(?:-(\d{1,2}))?\b")
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")
_NAMED_DATE = re.compile(
    r"\b(?:(\d{1,2})(?:st|nd|rd|th)?[ \t]+)?(" + _MONTH_WORD + r")\.?[ \t]*"
    r"(?:(\d{1,2})(?:st|nd|rd|th)?)?,?[ \t]*(\d{4})\b", re.I)


def _date_key(year: str | int, month: str | int, day: str | int | None) -> str | None:
    """A date as YYYY-MM or YYYY-MM-DD, or None when it is not one."""
    year, month = int(year), int(month)
    if not 1 <= month <= 12 or not 1000 <= year <= 9999:
        return None
    if not day:
        return f"{year:04d}-{month:02d}"
    day = int(day)
    return f"{year:04d}-{month:02d}-{day:02d}" if 1 <= day <= 31 else None


def _dates_in_text(text: str) -> set[str]:
    """Every date the text writes, normalised so two spellings of one date agree."""
    found = set()
    for year, month, day in _ISO_DATE.findall(text or ""):
        found.add(_date_key(year, month, day))
    for first, second, year in _NUMERIC_DATE.findall(text or ""):
        # Which of the two leading fields is the month is a convention this
        # module cannot know, so the one that can only be a month is taken as
        # one and the ambiguous case is read in the order the extractor writes.
        month, day = (first, second) if int(first) <= 12 else (second, first)
        found.add(_date_key(year, month, day))
    for before, name, after, year in _NAMED_DATE.findall(text or ""):
        found.add(_date_key(year, _MONTH_NUMBER[name.casefold()[:3]],
                            before or after))
    return {key for key in found if key}


def _collapse_dates(keys: set[str]) -> set[str]:
    """One entry per occasion.

    A month-precision date that a day-precision date already falls inside is
    that same occasion written less exactly, and counting the two separately
    would manufacture a second occasion out of one.
    """
    exact = {key for key in keys if len(key) > 7}
    return exact | {key for key in keys if len(key) == 7
                    and not any(e.startswith(key) for e in exact)}


def _dated_occasions(body: str, facts: list[dict],
                     documents: set[str]) -> tuple[int, int, int]:
    """Distinct dated occasions this element's block can be tested against.

    Returns the total, the number carried by the pages the block cites, and the
    number the block's own text writes, so the note printed beside the element
    can name which sources were consulted rather than announce a bare figure.

    Counting the dates the documents themselves carry is what replaces reading
    an adjective out of the allegation's wording: it measures the record rather
    than the phrasing, and it establishes a pattern exactly as readily as it
    fails to find one.

    Scoped to the pages the block cites, though. Counted across every fact
    routed to the allegation it counted the date of the memo that reported the
    conduct and the date of the interview that discussed it alongside the date
    of the conduct, so on any real corpus it reached two almost immediately and
    the repetition test never fired at all. What it measures now is narrower and
    honest: how many distinct dated facts sit on the pages this element was
    weighed on. It is still not a count of occasions of the conduct - a page can
    carry a fact dated for another reason - so it is used only to leave an
    element standing, never on its own to fail one.

    Two sources, because one of them is routinely empty for a reason that has
    nothing to do with the conduct: the extraction prompt tells the model to
    leave event_date null for a boundary, an approximation or an undated
    statement, so a page can carry the occasions and carry no date this module
    can count. The dates the block writes in its own prose are read as well, and
    an element is left unweighed only when NEITHER source yields anything. Both
    sources can only add occasions, and adding occasions can only leave a block
    standing on the verdict it wrote for itself, whichever way that reads.
    """
    # The citations are taken out before the block's own prose is read for
    # dates, because a filename can carry a month and a year of its own and a
    # document named for a month is not an occasion of the conduct.
    dates = _dates_in_text(_CITE_ANY.sub(" ", body))
    pages = _cited_pages(body, documents)
    on_pages = set()
    if pages:
        for fact in facts:
            stem = _normal_words(
                re.sub(r"\.[A-Za-z0-9]{1,5}$", "", str(fact.get("source_file") or "")))
            page = str(fact.get("source_page") or "").strip()
            when = str(fact.get("event_date") or "").strip()
            if not stem or not page.isdigit() or not when:
                continue
            if (stem, int(page)) in pages:
                # Normalised where it can be, so a date the extractor wrote in
                # one form and the block wrote in another is one occasion.
                on_pages |= _dates_in_text(when) or {when.casefold()}
    return (len(_collapse_dates(dates | on_pages)),
            len(_collapse_dates(on_pages)), len(_collapse_dates(dates)))


# What a MET value is rewritten to when the block cannot be weighed as written.
# Deliberately not "No": the tests below can find that an element's support is
# unreadable, and an unreadable support is a failure to weigh rather than a
# finding that the element fails. It carries no yes/no token, so re-reading the
# section afterwards finds the element unweighed rather than met either way.
_MET_UNACCEPTED = "Not accepted - see the note below"


def _correct_element_rows(section: str, rows: list[dict], elements: list[dict],
                          facts: list[dict],
                          documents: set[str]) -> tuple[str, list[dict], list[str]]:
    """Correct or withdraw an element verdict the block itself cannot carry.

    Two tests, both arithmetic on what the section already says rather than a
    second reading of the evidence, which is why they edit the written block
    instead of asking again. Each of them has an outcome in both directions:
    every one of them can leave an element standing, and the report says which
    happened.

    Repetition. An element the decomposition marked MULTIPLICITY: yes asserts
    that the conduct happened more than once, so one occasion does not meet it -
    that is what the element means, not a threshold chosen for it. Where the
    block states a count, the occasions established are the larger of that count
    and the distinct dated occasions the block cites or writes, and fewer than
    two of them records the element as not met. Where the block states no
    readable count at all and neither source yields two dated occasions, there
    is nothing to do the arithmetic on: the verdict is withdrawn and the element
    is left unweighed rather than failed, because a line the model did not write
    is not evidence that the conduct happened once.

    This arithmetic is asked only of a block weighed Yes, because what it tests
    is whether an affirmative finding of repeated conduct is carried by the
    record. Run against a block that already says No it could only restate the
    No, which is not a check.

    Citation. Every factual statement in this report cites its page. A block
    that cites no page, or whose only citations name no document that was
    ingested, has not been shown to rest on the corpus - but nor has it been
    shown to be wrong, and a citation this parser could not tie to a document is
    as likely to be an abbreviation as an invention. So these withdraw the
    verdict and name what could not be checked; they never record a finding of
    their own.

    That test is put to both answers. It used to examine only the blocks weighed
    Yes, on the reasoning that the burden sits on the allegation and an element
    the evidence does not carry needs nothing behind it. That reasoning says who
    has to prove what; it is not a reason to leave one of the two answers
    unchecked. What it produced was a MET: No resting on a document nobody
    ingested printed as a fully weighed element with no review marker, while the
    identical invention behind a MET: Yes was caught, annotated and withdrawn -
    so of the two answers, the one reachable without evidence was the adverse
    one. Withdrawing an unchecked No does not turn it into a Yes: coverage
    becomes incomplete, the table entails nothing, and the loud procedural
    default applies with the elements it could not weigh named beside it. What
    it stops is an unweighed answer of either kind being reported, and scored,
    as a weighed one.

    Only blocks naming an element that was actually planned are touched. A block
    numbered for an element nobody planned is logged and left alone, because
    correcting it would treat it as part of a table it is not part of.

    Both rewrites happen in one sweep ordered back to front by position in the
    text. Ordering by element number, which is what this used to do, holds only
    while the model writes its blocks in ascending order: out of order, editing
    one block moves every span recorded for the blocks above it, and the next
    edit lands in the middle of a word.
    """
    planned = {e["n"] for e in elements}
    repeated = {e["n"] for e in elements if e["multiplicity"]}
    notes: list[str] = []
    # (row, the verdict it is left with - False for not met, None for withdrawn,
    # the note that says which and why)
    corrections: list[tuple[dict, bool | None, str]] = []
    for row in rows:
        if row["n"] not in planned or row["met"] is None or not row["met_span"]:
            continue
        if row["met"] and row["n"] in repeated:
            dated, on_pages, in_text = _dated_occasions(row["body"], facts,
                                                        documents)
            stated = row["instances"]
            sources = (f"the pages it cites carry {on_pages} distinct dated "
                       f"fact(s) and its own text writes {in_text} distinct "
                       f"date(s)")
            if stated is None and dated < 2:
                # A block that wrote no count and a block whose count could not
                # be read are different failures, and the operator who reads
                # this note is the one who has to tell them apart.
                written = bool(_INSTANCES_LINE.search(row["body"])
                               or _INSTANCES_LOOSE.search(row["body"]))
                count_clause = ("The count on its INSTANCES ESTABLISHED line "
                                "could not be read as a number" if written else
                                "Its block states no count of occasions")
                corrections.append((row, None, (
                    f"*Element E{row['n']} asserts that the conduct happened on "
                    f"more than one occasion. {count_clause}, and {sources}, so "
                    f"there is nothing here to test that assertion against in "
                    f"either direction. The verdict written for this element is "
                    f"not accepted and the element is left unweighed. This "
                    f"records what could not be weighed, not a finding that the "
                    f"conduct happened once.*")))
                log.warning("element E%d asserts repetition but its block states "
                            "no readable count and offers %d dated "
                            "occasion(s); verdict withdrawn", row["n"], dated)
                continue
            if stated is not None and max(stated, dated) < 2:
                corrections.append((row, False, (
                    f"*Element E{row['n']} asserts that the conduct happened on "
                    f"more than one occasion. The count stated was {stated}, and "
                    f"{sources}, so more than one occasion is not established "
                    f"and the element is recorded as not met.*")))
                log.warning("element E%d asserts repetition; stated count %r and "
                            "%d dated occasion(s) cited or written; forced to "
                            "MET: No", row["n"], stated, dated)
                continue
        # The same words for either answer, so that the note beside a withdrawn
        # No reads as the same kind of correction as the note beside a withdrawn
        # Yes, which is what it is.
        weighed_as = "met" if row["met"] else "not met"
        cited = [c for c in _CITE_ANY.findall(row["body"])
                 if _carries_page(c, documents)]
        grounded = [c for c in cited if _names_document(c, documents)]
        if not cited:
            corrections.append((row, None, (
                f"*Element E{row['n']} was weighed as {weighed_as} but its block "
                f"carries no citation to a document and a page, so what the "
                f"verdict rests on cannot be checked. The verdict is not "
                f"accepted and the element is left unweighed. This records a "
                f"failure to weigh, not a finding about the evidence.*")))
            log.warning("element E%d weighed %s with no page-bearing citation; "
                        "verdict withdrawn", row["n"], weighed_as)
        elif not grounded:
            named = ", ".join(" ".join(c.split()) for c in cited[:2])
            corrections.append((row, None, (
                f"*Element E{row['n']} was weighed as {weighed_as} on {named}, "
                f"which names no document in this corpus. A citation to a record "
                f"nobody ingested cannot be checked, so the verdict is not "
                f"accepted and the element is left unweighed. This records a "
                f"failure to weigh, not a finding about the evidence.*")))
            log.warning("element E%d weighed %s on citation(s) naming no "
                        "ingested document (%s); verdict withdrawn",
                        row["n"], weighed_as, named)

    # Back to front by position, so an earlier edit cannot move a later span.
    for row, verdict, _note in sorted(corrections,
                                      key=lambda c: c[0]["met_span"][0],
                                      reverse=True):
        start, end = row["met_span"]
        section = section[:start] + ("No" if verdict is False
                                     else _MET_UNACCEPTED) + section[end:]
        row["met"] = verdict
    notes.extend(note for _row, _verdict, note in
                 sorted(corrections, key=lambda c: c[0]["n"]))
    return section, rows, notes


# The disposition line, matched through the same tolerant pattern wherever it is
# read, inserted before, or replaced. A rewriter insisting on "**Disposition:**"
# while the model wrote "**Disposition**:" found nothing, appended a second and
# contradictory line at the foot of the section, and left the dispositions table
# reporting one label while the section it claims to be assembled from showed
# another.
# A dash separates a label from its heading as readily as a colon does, and a
# line the rewriter cannot find is a line it appends a second, contradictory
# copy of.
_DISPOSITION_LINE = re.compile(
    _anchored(r"Disposition[ \t]*[:\-–—][ \t]*(.*)$"), re.I | re.M)
# A qualifier is not a label: "Substantiated in part" is a claim about which
# elements were carried and belongs in the element table, not in the enum. The
# bare word "only" used to sit in this list and cost the module a correct label
# whenever the model attached its reason in the same breath - "Not substantiated
# - only one instance was established" was read as unreadable and the report
# then accused the model of writing an impermissible disposition it had not
# written. A partiality qualifier names what was carried; "only" on its own does
# not, and where it does it says "only as to", which the list still catches.
_PARTIAL = re.compile(r"\b(?:in part|partly|partially|as to|except)\b")


def _disposition_match(section: str) -> tuple[re.Match | None, _Markup]:
    """The disposition line of a section, found on its decoration-free copy."""
    markup = _Markup(section)
    return _DISPOSITION_LINE.search(markup.text), markup


def _note_before_disposition(section: str, notes: list[str]) -> str:
    """Put a correction note with the element block it corrects.

    Above the disposition rather than at the foot of the section: a reader who
    meets the note only after the findings has already read a MET line that the
    generator had by then decided was wrong.
    """
    block = "\n".join(notes)
    match, markup = _disposition_match(section)
    if match:
        at = markup.raw_line_start(match.start())
        return section[:at] + block + "\n\n" + section[at:]
    return f"{section.rstrip()}\n\n{block}\n"


def _weighed_elements(rows: list[dict], elements: list[dict]) -> dict[int, bool]:
    """Element number -> verdict, for blocks naming an element that was planned.

    A block numbered for an element nobody planned does not vote. It is a
    hallucinated or renumbered block, and letting it vote is what allowed a
    table weighing E2, E4 and E5 to reach the same total as a complete one while
    E1 went unexamined.

    Two blocks numbered for the same element and disagreeing do not vote either.
    Keyed by element number, the later block simply overwrote the earlier one,
    so the table's verdict for an element was whichever copy of it the model
    wrote last. Disagreeing copies are a table that entails nothing about that
    element, which is what the caller is told.
    """
    planned = {e["n"] for e in elements}
    seen: dict[int, set[bool]] = {}
    for row in rows:
        if row["met"] is None or row["n"] not in planned:
            continue
        seen.setdefault(row["n"], set()).add(bool(row["met"]))
    weighed = {}
    for number in sorted(seen):
        verdicts = seen[number]
        if len(verdicts) == 1:
            weighed[number] = next(iter(verdicts))
        else:
            log.warning("element E%d is weighed both ways by different blocks in "
                        "the same section; it is counted as unweighed", number)
    return weighed


def _unweighed_elements(rows: list[dict], elements: list[dict]) -> list[int]:
    """The planned element numbers that no readable block weighed."""
    weighed = set(_weighed_elements(rows, elements))
    return sorted({e["n"] for e in elements} - weighed)


def _stray_rows(rows: list[dict], elements: list[dict]) -> list[int]:
    """Block numbers in the section that name no planned element."""
    planned = {e["n"] for e in elements}
    return sorted({r["n"] for r in rows if r["n"] not in planned})


def _required_disposition(rows: list[dict], elements: list[dict],
                          labels: tuple[str, str] = DISPOSITIONS) -> str | None:
    """What the element table entails, or None when it does not entail anything.

    Coverage is tested by identity, not by arithmetic: the numbers weighed have
    to BE the numbers planned. Comparing counts let a table that invented E4 and
    E5 while skipping E1 reach the length of a complete one and entail
    Substantiated on an element that was never looked at.

    A partial table can still entail Not substantiated - one failed element is
    enough, whatever else went unweighed - but it can never manufacture a
    Substantiated out of the elements the model happened to write down.
    """
    weighed = _weighed_elements(rows, elements)
    if not weighed:
        return None
    if any(met is False for met in weighed.values()):
        return labels[1]
    if elements and set(weighed) == {e["n"] for e in elements}:
        return labels[0]
    return None


# Written where the label is read rather than beside the enum, because these are
# what a model reaches for when it is asked for one of two labels and writes a
# near neighbour of it. They map onto whichever pair is in force; they are not
# themselves labels and are never printed.
_SYNONYMS_NEGATIVE = re.compile(
    r"^(?:not substantiated|unsubstantiated|not sustained|unsustained|"
    r"not substantiat\w*|not sustain\w*)\b")
_SYNONYMS_AFFIRMATIVE = re.compile(r"^(?:substantiat\w*|sustain\w*)\b")
# Where the label stops and the model's reason for it begins. A comma is not on
# this list: "Substantiated, in part" is a qualified label rather than a label
# with a reason, and splitting on the comma would promote it to the bare one.
_LABEL_CLAUSE = re.compile(r"[-–—;(]")


def _normalize_disposition(raw: str | None,
                           labels: tuple[str, str] = DISPOSITIONS) -> str | None:
    """One of the two labels of record, or None for anything else.

    A partiality qualifier sends the label back as unreadable rather than being
    matched past. Prefix-matching promoted "Substantiated in part" - a model
    correctly noticing it had carried two of three elements - to a full adverse
    finding against the subject, while "Partially substantiated" was already
    being rejected, so one intent was handled two opposite ways.

    The qualifier is looked for in the label clause only. A fixed four-word
    window took in the start of the model's reason as well, so a plain label
    with its reason attached in the same breath came back unreadable and the
    section was then annotated to say the model had written an impermissible
    disposition - in the one channel a reviewer uses to audit these corrections.
    """
    clause = _LABEL_CLAUSE.split(raw or "", 1)[0]
    text = " ".join(re.sub(r"[^a-z ]+", " ", clause.casefold()).split())
    if not text or _PARTIAL.search(text):
        return None
    # The negative label is tested first: it usually contains the affirmative
    # one, so testing the affirmative first would match the wrong half of it.
    negative = _normal_words(labels[1])
    affirmative = _normal_words(labels[0])
    if negative and re.match(re.escape(negative) + r"\b", text):
        return labels[1]
    if _SYNONYMS_NEGATIVE.match(text):
        return labels[1]
    if affirmative and re.match(re.escape(affirmative) + r"\b", text):
        return labels[0]
    if _SYNONYMS_AFFIRMATIVE.match(text):
        return labels[0]
    return None


def _rewrite_disposition(section: str, label: str, note: str) -> str:
    """Replace the first disposition line, which is the one every reader parses."""
    replacement = f"**Disposition:** {label}\n\n*{note}*"
    match, markup = _disposition_match(section)
    if match:
        start = markup.raw_line_start(match.start())
        end = markup.origin(match.end())
        return section[:start] + replacement + section[end:]
    return f"{section.rstrip()}\n\n{replacement}\n"


def _coverage_gap(missing: list[int], stray: list[int]) -> str:
    """What the element table left unweighed, in one clause, or "".

    The same sentence whether the table went on to entail a disposition or not.
    An entailed disposition used to silence it, so an allegation settled on one
    readable failed element out of five read exactly like an allegation whose
    whole table had been weighed, while the same degree of parse failure was
    announced loudly whenever it happened to land the other way.
    """
    parts = []
    if missing:
        parts.append("element" + ("s " if len(missing) > 1 else " ") +
                     ", ".join(f"E{n}" for n in missing) +
                     (" were" if len(missing) > 1 else " was") + " never weighed")
    if stray:
        parts.append("block(s) " + ", ".join(f"E{n}" for n in stray) +
                     " name no element that was planned")
    return "; ".join(parts)


def _settle_disposition(section: str, rows: list[dict], elements: list[dict],
                        index: int,
                        labels: tuple[str, str] = DISPOSITIONS
                        ) -> tuple[str, str, str]:
    """Bind the disposition to the element table, or say plainly that it is not.

    The table is the authority. It is the model's own weighing, so a disposition
    disagreeing with it is a slip rather than a judgement and is corrected
    without asking again, and every correction is written into the section as a
    note: a repaired disposition that leaves no trace is indistinguishable from
    a model that got it right.

    The branch that mattered was the other one. A table entailing nothing -
    unreadable, or covering only some of the elements that were planned - used
    to hand the decision back to whatever label the model had written by hand,
    which meant the whole element machinery could be stepped around by leaving
    an element block out. It cannot now. The section has already been written a
    second time by the caller when the table could not be read, so reaching here
    with nothing entailed is a persistent failure of generation and is recorded
    as one: an element nobody weighed has not been proved, the burden sits on
    the allegation, and the section names the elements left unweighed so that a
    reader sees the disposition rests on a failure to weigh rather than on the
    record. That is a statement about the generation, not a finding about the
    evidence, and it says so where nobody can miss it.

    A disposition the table did entail is returned with the same coverage clause
    attached whenever anything went unweighed. The default of record is only
    ever reached procedurally, and it has to be as visible when the arithmetic
    happens to agree with it as when it does not.

    Returns the section, the disposition, and the reason the allegation needs an
    operator's eye, or "" when the table settled the question on its own.
    """
    match, _markup = _disposition_match(section)
    written = match.group(1).strip().strip("* ") if match else ""
    stated = _normalize_disposition(written, labels)
    required = _required_disposition(rows, elements, labels)

    missing = _unweighed_elements(rows, elements)
    stray = _stray_rows(rows, elements)
    if stray:
        log.warning("allegation %d: element block(s) %s name no planned element "
                    "and were not counted toward coverage",
                    index, ", ".join(f"E{n}" for n in stray))

    if required is not None:
        gap = _coverage_gap(missing, stray)
        review = ""
        if gap:
            review = (f"{gap}, so the disposition rests on part of the element "
                      f"table only")
            log.warning("allegation %d: disposition %r entailed while %s",
                        index, required, gap)
        if stated == required:
            return section, required, review
        if stated is None:
            missing_line = ("the section carried no disposition line" if not match
                            else f'"{written}" is not a permitted disposition')
            added = ("A disposition line has been added below."
                     if not match else
                     "The line has been replaced with the disposition of record.")
            note = (f"Disposition corrected at generation: {missing_line}. The "
                    f"burden sits on the allegation, so this is recorded as "
                    f"{required}. {added}")
        else:
            unmet = [str(n) for n, met in
                     sorted(_weighed_elements(rows, elements).items())
                     if met is False]
            because = (f"element{'s' if len(unmet) > 1 else ''} "
                       f"E{', E'.join(unmet)} "
                       f"{'were' if len(unmet) > 1 else 'was'} not met"
                       if unmet else "every element was met")
            note = (f'Disposition corrected at generation: {because}, so the '
                    f'allegation is {required}; the model wrote "{stated}".')
        log.warning("allegation %d: disposition %r corrected to %r",
                    index, written, required)
        return _rewrite_disposition(section, required, note), required, review

    # "No block could be read" and "no verdict in the blocks could be read" are
    # different failures, and the reason printed here is what the operator is
    # pointed at and what the retry was told to fix. Reported as the first when
    # it was the second, the operator is sent to look at a table that is there
    # and the model is asked to correct block coverage that was never wrong.
    planned_rows = [row for row in rows if row["n"] in {e["n"] for e in elements}]
    if len(missing) < len(elements):
        reason = _coverage_gap(missing, [])
    elif not rows:
        reason = ("no element block in the section could be read, so no element "
                  "was weighed")
    elif not planned_rows:
        reason = _coverage_gap([], stray) + ", so no element was weighed"
    else:
        reason = (f"{len(planned_rows)} element block(s) were read but no "
                  f"verdict in them could be read or accepted, so no element "
                  f"was weighed")
    gap = _coverage_gap([], stray)
    if gap and planned_rows:
        # Only where it adds something: the branch above is that clause already.
        reason += f" ({gap})"
    added = ("A disposition line has been added below."
             if not match else
             "The line has been replaced with the disposition of record.")
    note = (f"DISPOSITION UNSETTLED - OPERATOR REVIEW REQUIRED. "
            f"{reason[0].upper()}{reason[1:]}, so the element table entails "
            f"nothing and the label written in the section" +
            (f' ("{written}")' if written else "") +
            f" has not been accepted: accepting it would let an allegation be "
            f"disposed of by leaving an element out. An element that was not "
            f"weighed has not been proved and the burden sits on the "
            f"allegation, so the disposition of record is {labels[1]}. This "
            f"records a failure to weigh the allegation, not a finding about "
            f"the evidence, and it must not be read as one. {added}")
    log.error("allegation %d: %s; recorded %s pending operator review",
              index, reason, labels[1])
    return (_rewrite_disposition(section, labels[1], note), labels[1], reason)


# Both headings carry the same tail as every other heading in this module. Ended
# on the word itself, a heading that gave itself a title - "Conflicts
# identified", "CORROBORATION AND AGREEMENT:" - matched nothing, and a body
# whose corroboration heading went unmatched was promoted whole to the candidate
# conflicts the findings pass is instructed to adjudicate: a list of accounts
# that agree, handed over as disagreements to be resolved.
_HEADING_CONFLICTS = re.compile(_anchored(r"CONFLICTS\b" + _HEADING_TAIL),
                                re.I | re.M)
_HEADING_CORROBORATION = re.compile(
    _anchored(r"CORROBORATION\b" + _HEADING_TAIL), re.I | re.M)
# Every conflict candidate the template asks for opens with a TYPE line, so its
# presence is what tells an unlabelled body apart from a corroboration list.
_TYPE_LINE = re.compile(r"\bTYPE[ \t]*:[ \t]*[a-z-]{3,}", re.I)


def _split_candidates(text: str, documents: set[str] | None = None) -> tuple[str, str]:
    """The conflict pass's output, separated into conflicts and corroboration.

    Both headings are matched through the markdown tolerance, because the model
    writes "**CONFLICTS:**" and "### CONFLICTS:" as freely as the bare form, and
    when neither matched the whole body used to be promoted to conflicts. That
    turned a corroboration list - a list of accounts that agree - into the
    candidate conflicts the findings pass is told to adjudicate, priming it that
    a disagreement exists where the comparison had found none.

    So an unlabelled body is now read for the shape the template asks conflicts
    to take. One carrying a TYPE line is still treated as conflicts in full,
    because an unexpected shape should cost tidiness rather than recall; one
    carrying none goes to corroboration, where an agreement belongs and where
    nothing tells the findings pass to adjudicate it.
    """
    markup = _Markup((text or "").strip())
    body = markup.text
    if not body.strip() or body.strip().upper() == "NONE":
        return "NONE", "NONE"
    end = len(body)
    conflicts, corroboration = markup.raw, ""
    con = _HEADING_CONFLICTS.search(body)
    cor = _HEADING_CORROBORATION.search(body)
    if con and cor:
        if cor.start() > con.start():
            conflicts = markup.raw_block(con.end(), cor.start())
            corroboration = markup.raw_block(cor.end(), end)
        else:
            corroboration = markup.raw_block(cor.end(), con.start())
            conflicts = markup.raw_block(con.end(), end)
    elif con:
        conflicts = markup.raw_block(con.end(), end)
    elif cor:
        conflicts = markup.raw_block(0, cor.start())
        corroboration = markup.raw_block(cor.end(), end)
    elif not _TYPE_LINE.search(body):
        log.warning("the conflict pass returned neither heading and no TYPE "
                    "line; the body is treated as corroboration, not conflicts")
        conflicts, corroboration = "NONE", markup.raw
    return (_filter_conflicts(conflicts, documents or set()),
            (corroboration.strip() or "NONE"))


def _marker_column(match: re.Match) -> int:
    """How far a list marker is indented from the margin."""
    marker = match.group(0)
    return len(marker) - len(marker.lstrip(" \t"))


def _split_items(body: str) -> list[str]:
    """One entry per candidate, numbered or bulleted.

    Numbering is tried first and bullets only when nothing was numbered, so a
    model that bullets the sub-lines of a numbered entry does not have that
    entry split into pieces.

    That order covers a model that bullets its sub-lines; it does not cover one
    that NUMBERS them, and a numbered sub-line split one candidate into two - the
    first of them then stating no incompatibility, which the filter below reads
    as a candidate that is not a conflict and drops. So only markers at the
    column of the first one split the body; a marker indented under an entry is
    part of that entry.
    """
    for pattern in (_NUMBERED_ITEM, _BULLET_ITEM):
        found = list(pattern.finditer(body))
        if not found:
            continue
        column = _marker_column(found[0])
        starts = [m for m in found if _marker_column(m) == column]
        items = []
        for i, match in enumerate(starts):
            end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
            piece = body[match.end():end].strip()
            if piece:
                items.append(piece)
        if items:
            return items
    return []


def _filter_conflicts(block: str, documents: set[str] | None = None) -> str:
    """Drop candidates that the item's own shape shows are not conflicts.

    Only two shapes are dropped, both of them structural: a contradiction or a
    wording variance whose two sides cite the same page is one source quoted
    twice rather than two accounts disagreeing, and a contradiction that states
    no incompatibility has not met the second half of the test it was given.
    Observation limits and contradicted defences legitimately cite one source,
    so they are never dropped, and if the filter would empty the list it keeps
    every item with the objection attached to it rather than re-admitting them
    clean - the findings pass adjudicates each candidate, and it should see the
    objection at the same time as the candidate rather than a list the filter
    had privately decided against.

    `documents` is the corpus, and it is here so that the page a citation
    carries is read the same way here as everywhere else in this module: a
    citation written "[file:7]" carries page seven for the element table, and a
    filter that could not see it would decide two sides cite different pages
    because it could read neither of them.
    """
    body = (block or "").strip()
    if not body or body.upper().startswith("NONE"):
        return "NONE"
    items = _split_items(body)
    if not items:
        log.warning("the conflict candidates carry no item marker; they are "
                    "passed through unfiltered")
        return body

    kept, dropped = [], []
    for item in items:
        # The item's own fields are read off its decoration-free copy, so
        # '**TYPE:** **contradiction**' is the same item as the plain form. Its
        # citations are read off the item itself, because a citation stripped of
        # punctuation no longer names the document it cites.
        plain = _Markup(item).text
        found = re.search(r"TYPE[ \t]*:[ \t]*([a-z-]+)", plain, re.I)
        kind = found.group(1).lower() if found else ""
        reason = ""
        if kind in ("contradiction", "wording-variance"):
            cites = [c for c in _CITE.findall(item)
                     if _carries_page(c, documents or set())]
            distinct = {" ".join(c.split()).casefold() for c in cites}
            if len(cites) >= 2 and len(distinct) < 2:
                reason = "both sides cite the same page"
            elif kind == "contradiction" and not re.search(
                    r"INCOMPATIBLE[ \t]*:[ \t]*\S", plain, re.I):
                reason = "no stated incompatibility between the two accounts"
        (dropped if reason else kept).append((item, reason))

    if not kept:
        log.warning("the conflict filter objects to all %d candidate(s); they "
                    "are kept for adjudication with the objection attached",
                    len(items))
        kept = [(f"{item}\n   FILTER OBJECTION: {reason} — say whether this is a "
                 f"conflict at all before adjudicating it.", reason)
                for item, reason in dropped]
        dropped = []
    for item, reason in dropped:
        log.info("dropped conflict candidate (%s): %s", reason,
                 " ".join(item.split())[:120])
    return "\n\n".join(f"{i}. {item}" for i, (item, _r) in enumerate(kept, 1))


_FINDINGS_HEAD = re.compile(_anchored(r"Findings\b" + _HEADING_TAIL), re.I | re.M)
# The same tolerance as the head above it and as _SECTION_BREAK. Written without
# the tail, this closed the findings block on any FINDING that happened to open
# with one of these words - "Conflicts between the two accounts were resolved in
# favour of the log [memo.pdf p.2]" is a finding, not a heading - and the
# summary was then told the section carried no findings at all. _HEADING_TAIL
# refuses a line carrying a bracket or more than three trailing words, so a real
# heading still closes the block.
_FINDINGS_END = re.compile(
    _anchored(r"(?:Conflicts|Gaps|Corroboration|Disposition)\b" + _HEADING_TAIL),
    re.I | re.M)


def _findings_block(section: str) -> str:
    """The Findings list out of a written section, for the summary pass.

    The summary is asked for the one reason behind each disposition together
    with its citation, so it has to be shown the findings it is citing from.
    Given only the allegation text and its verdict it wrote a plausible
    [file p.N] anyway, at the top of the report, where it is the first thing
    read and the least grounded thing in it.

    A heading ending in a colon - ordinary markdown - used to defeat this, and
    the summary was then told the section carried no findings and instructed to
    write that the disposition rests on no cited finding, above a section full
    of cited findings.
    """
    markup = _Markup(section)
    head = _FINDINGS_HEAD.search(markup.text)
    if not head:
        return "(this section recorded no findings)"
    end = _FINDINGS_END.search(markup.text, head.end())
    stop = end.start() if end else len(markup.text)
    return markup.raw_block(head.end(), stop).strip() or \
        "(this section recorded no findings)"


def _basis(disposition: dict) -> str:
    """What the disposition rests on, printed in the same row as the disposition.

    A label on its own cannot tell an allegation the records refute from one the
    corpus never spoke to; both print as the same word in the same typeface. The
    counts go beside it - how much of the element table was actually weighed and
    how many statements in the section cite a page of a document that was
    ingested - so that a verdict resting on nothing is visible without reading
    the section it came from.
    """
    counts = (f"{disposition['weighed']} of {disposition['planned']} element(s) "
              f"weighed, {disposition['citations']} cited statement(s)")
    if disposition["review"]:
        return f"**REVIEW REQUIRED** — {disposition['review']}; {counts}"
    return counts


def generate(goal: str | None = None, allegations: list[str] | None = None, *,
             allow_incomplete: bool = False, seed: int | None = None):
    """Yield ('status'|'token'|'error'|'done', payload) while writing the report.

    allow_incomplete is an explicit operator override for the corpus-integrity
    refusal. It is keyword-only so the existing positional call sites keep
    working, and it defaults off because the refusal exists precisely for the
    case where nobody has noticed the corpus is short.

    seed fixes the sampler for this run. Left None the module's default seed
    applies and the same corpus produces the same report, which is what makes a
    change in the output attributable to a change in the code. Passed a seed, a
    caller can draw several independent reports from one corpus and compare
    them, which is the only way to see which findings are stable and which are
    an artefact of one draw.
    """
    goal = (goal if goal is not None else get_goal()).strip()
    allegations = allegations if allegations is not None else get_allegations()
    allegations = [a.strip() for a in allegations if a.strip()]
    if not allegations:
        yield "error", "No allegations have been entered. Add at least one."
        return

    # The name follows the mode, so that a report written against the
    # operator's endpoint asks it for a model that endpoint serves.
    model = llm_settings.effective_text_model()
    if not model:
        yield "error", ("No text model is set, so no report can be written. "
                        "Choose one on the settings page, or set TEXT_MODEL "
                        "in .env and restart.")
        return

    docs = evidence.corpus_state()
    facts_total = state.query_one("SELECT COUNT(*) AS n FROM triples")
    if not facts_total or not facts_total["n"]:
        yield "error", ("No facts have been extracted yet. Upload documents and "
                        "let them finish processing first.")
        return

    # The corpus invariant, settled before any model work: a report that names a
    # gap it could have filled from a document it claims to have read is worse
    # than no report, and the reader cannot tell the two apart from the page.
    blocking = evidence.blocking_faults(docs)
    if blocking and not allow_incomplete:
        detail = "; ".join(f"{b['filename']} ({b['reason']})" for b in blocking)
        log.error("report refused: %s", detail)
        yield "error", (
            f"Report refused: {detail}. A report cannot be written from a corpus "
            f"that is still processing or that has lost assertions it recorded - "
            f"findings would rest on evidence the analysis never saw. Let the "
            f"queue finish, or re-ingest the documents named, then try again.")
        return

    # The labels of record for this investigation, read once so that every
    # prompt, every stored verdict and the dispositions table use the same pair.
    labels = disposition_labels()
    system = SYSTEM_TEMPLATE.format(yes_label=labels[0], no_label=labels[1])

    # Named before it happens: on a cold endpoint this call is the pull of a
    # multi-gigabyte model, and it sits between the click and the first
    # allegation with nothing else to show for the wait.
    yield "status", f"Checking {model} is loaded"
    client = Ollama()
    try:
        client.require_model(model, llm_settings.text_model_label())
    except Exception as exc:
        yield "error", str(exc).splitlines()[0]
        return

    options = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                              "REPORT_NUM_PREDICT", 1400, seed=seed)
    # The decomposition answers with a short table and nothing else, so it does
    # not need the findings budget and should not spend it.
    short_options = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                                    "ELEMENTS_NUM_PREDICT", 500, seed=seed)
    # A second attempt at a pass that returned something unreadable has to be an
    # independent draw. The same seed at the same temperature reproduces the
    # same text, so retrying without changing it would return the identical
    # unreadable answer and cost a call to learn nothing.
    retry_options = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                                    "REPORT_NUM_PREDICT", 1400, seed=random_seed())
    retry_short = default_options("TEXT_TEMPERATURE", "TEXT_NUM_CTX",
                                  "ELEMENTS_NUM_PREDICT", 500, seed=random_seed())

    index_map = evidence.evidence_index(len(allegations))
    speakers = evidence.speakers()
    manifest = _manifest_block(docs)
    speaker_note = _speakers_block(docs, speakers)
    # The corpus as a set of names, for testing whether a bracket cites one of
    # these documents or cites the allegation back at itself.
    document_names = _document_names(docs)

    # Every allegation's evidence is gathered before a word is written. All of
    # it is local, and knowing which documents contributed is what lets the
    # integrity warning appear above the findings rather than after them.
    contexts: list[dict] = []
    for index, allegation in enumerate(allegations, 1):
        yield "status", f"Allegation {index} of {len(allegations)}: gathering evidence"
        # Retrieve wide, then route, then trim. Routing is a filter, so taking
        # the top MAX_PASSAGES first and filtering afterwards leaves however
        # many survive - often far fewer than the pass needs, and the shortfall
        # falls hardest on the conflict pass, which cannot report a
        # contradiction whose two halves were never both retrieved.
        passages = embed.search(allegation, k=MAX_PASSAGES * RETRIEVAL_WIDTH)
        facts = evidence.enrich(chat.relationships_for(allegation, passages), index_map)
        passages, facts = evidence.route(passages, facts, index, index_map)
        passages = passages[:MAX_PASSAGES]
        contexts.append({"index": index, "allegation": allegation,
                         "passages": passages, "facts": facts})

    # The corpus is read again now that retrieval is finished. A report is
    # several model passes per allegation on local hardware, and a document
    # uploaded during the run is queued into a corpus this report has already
    # stopped looking at: it appears in no manifest, contributes to no
    # allegation, and is counted in no header line, while the report reads as
    # complete.
    settled = evidence.corpus_state()
    corpus_changed = ({(d["doc_id"], d.get("status")) for d in settled} !=
                      {(d["doc_id"], d.get("status")) for d in docs})
    if corpus_changed:
        log.error("the ingested document set changed while the report was being "
                  "written: %d document(s) at the start, %d now",
                  len(docs), len(settled))

    silent = evidence.silent_documents(docs, evidence.contributing(contexts))
    banner = _integrity_banner(blocking if allow_incomplete else [], silent,
                               len(docs), corpus_changed)
    if banner:
        log.error("corpus integrity: %d of %d ingested document(s) contributed "
                  "nothing to this report",
                  len({r["doc_id"] for r in silent + blocking}), len(docs))
        yield "token", banner

    body: list[str] = []
    dispositions: list[dict] = []

    for ctx in contexts:
        index, allegation = ctx["index"], ctx["allegation"]
        relationships = _format_relationships(ctx["facts"])
        passage_block = _format_passages(ctx["passages"], speakers)

        # The allegation is decomposed with no evidence in front of the model.
        # Shown the record first it reads the elements back off whatever the
        # record happens to admit, which is how one conceded act came to satisfy
        # a claim about a course of conduct.
        elements_prompt = ELEMENTS_TEMPLATE.format(allegation=allegation)
        elements, element_note = [], ""
        # The decomposition is what every check below is measured against, so an
        # answer with anything unreadable in it - no element blocks at all, or
        # blocks whose MULTIPLICITY line could not be read - is worth a second,
        # independently seeded call before the allegation is tested on it.
        for attempt in (1, 2):
            # Inside the loop rather than above it. Each attempt is a whole
            # model call with nothing streamed out of it, so an attempt that
            # does not announce itself is a second silence indistinguishable
            # from the first one still running.
            yield "status", (f"Allegation {index} of {len(allegations)}: "
                             f"breaking the allegation into elements"
                             + ("" if attempt == 1 else " (second attempt)"))
            raw_elements = ""
            try:
                answer = client.generate(
                    model, elements_prompt, system=system,
                    options=short_options if attempt == 1 else retry_short,
                    think=thinking_enabled())
                raw_elements = (answer.get("response") or "").strip()
            except Exception as exc:
                log.warning("element pass failed for allegation %d on attempt "
                            "%d: %s", index, attempt, exc)
            elements, element_note = _elements_for(allegation, raw_elements)
            if not element_note:
                break
            log.warning("allegation %d: the element pass returned something "
                        "unreadable on attempt %d", index, attempt)
        element_block = _elements_block(elements)

        # Conflict detection gets its own pass with nothing else to do. Asked
        # for alongside findings it rides on chance - one graded run caught a
        # wording conflict, the next demoted it to a gap. A single-purpose
        # comparison first, adjudicated inside the findings second, pins it.
        conflict_candidates, corroboration = "NONE", "NONE"
        conflict_draws, corroboration_draws = [], []
        for draw in range(CONFLICT_DRAWS):
            yield "status", (f"Allegation {index} of {len(allegations)}: "
                             f"comparing witnesses (draw {draw + 1} of "
                             f"{CONFLICT_DRAWS})")
            try:
                draw_options = dict(options)
                draw_options["seed"] = random_seed()
                draw_options["temperature"] = max(
                    float(options.get("temperature") or 0.0),
                    CONFLICT_TEMPERATURE)
                found = client.generate(
                    model,
                    CONFLICT_TEMPLATE.format(
                        allegation=allegation,
                        speakers=speaker_note,
                        relationships=relationships,
                        passages=passage_block),
                    system=system, options=draw_options,
                    think=thinking_enabled())
                one, two = _split_candidates(
                    found.get("response") or "", document_names)
                conflict_draws.append(one)
                corroboration_draws.append(two)
            except Exception as exc:
                log.warning("conflict pass %d failed for allegation %d: %s",
                            draw + 1, index, exc)
        conflict_candidates = _merge_candidates(conflict_draws)
        corroboration = _merge_candidates(corroboration_draws)
        log.info("allegation %d: %d comparison draw(s) pooled", index,
                 len(conflict_draws))

        yield "status", f"Allegation {index} of {len(allegations)}: writing findings"
        goal_note = (f"INVESTIGATIVE GOAL (context only - goals are answered, "
                     f"never 'substantiated'):\n{goal}\n\n" if goal else "")
        prompt = goal_note + ALLEGATION_TEMPLATE.format(
            number=index, allegation=allegation, elements=element_block,
            conflicts=conflict_candidates, corroboration=corroboration,
            speakers=speaker_note, manifest=manifest,
            relationships=relationships, passages=passage_block,
            yes_label=labels[0], no_label=labels[1])

        # A table that does not weigh every element it was given is a failure
        # to follow the shape rather than a judgement about the evidence, and
        # the answer to it is to ask again, with an independent seed and the
        # shape restated. Only once: a loop here would spend a local machine's
        # afternoon on an allegation.
        section, rows, notes, circular = "", [], [], 0
        retry_note = RETRY_NOTE_COVERAGE
        for attempt in (1, 2):
            section = ""
            try:
                for token in client.stream(
                        model, prompt if attempt == 1 else prompt + retry_note,
                        system=system,
                        options=options if attempt == 1 else retry_options,
                        think=thinking_enabled()):
                    section += token
                    yield "token", token
            except Exception as exc:
                log.error("allegation %d failed: %s", index, exc)
                yield "error", str(exc).splitlines()[0]
                return
            yield "token", "\n\n"

            # Everything below repairs the written section rather than the
            # streamed copy, so a correction is always visible in the stored
            # report even though the live view has already shown the model's
            # first answer.
            #
            # Circular citations go first, before the element blocks are
            # weighed. Scrubbing afterwards left an element recorded MET: Yes
            # beside a SUPPORTING line the scrubber had just emptied, with the
            # disposition already settled on it and nothing to revisit it.
            section, circular = _scrub_allegation_citations(section, document_names)
            section, foreign = _drop_foreign_findings(
                section, ctx["facts"], index, allegations)
            section = section.rstrip() + _secondhand_note(
                _secondhand_facts(index) or ctx["facts"])
            for line in foreign:
                log.info("allegation %d: dropped a finding belonging to another "
                         "allegation: %s", index, line)
            rows = _parse_element_rows(section)
            # Which half of the shape the parser could not read, taken here
            # rather than after the corrections below, which withdraw verdicts
            # that were themselves perfectly readable.
            planned_numbers = {e["n"] for e in elements}
            covered = planned_numbers <= {r["n"] for r in rows}
            unread = {r["n"] for r in rows if r["met"] is None} & planned_numbers
            section, rows, notes = _correct_element_rows(
                section, rows, elements, ctx["facts"], document_names)
            unweighed = _unweighed_elements(rows, elements)
            # The retry is gated on the table being COMPLETE, not on it
            # entailing something. Gated on entailment, a single readable
            # MET: No ended the retry however many of the other blocks had gone
            # unread, while the same degree of unreadability was retried and
            # then announced whenever it happened to point the other way. The
            # cost of asking again is one call; the cost of not asking is a
            # disposition settled on a table nobody could finish reading.
            if attempt == 2 or not unweighed:
                break
            # The retry restates whatever the parser actually failed on: the
            # blocks, the verdict line inside them, or what the verdict rested
            # on. Told to fix its block coverage when the coverage had been
            # right, the model reproduced the same section on the second draw
            # and the allegation banked the procedural default.
            if not covered:
                retry_note = RETRY_NOTE_COVERAGE
            elif unread:
                retry_note = RETRY_NOTE_MET
            else:
                retry_note = RETRY_NOTE_SUPPORT
            log.error("allegation %d: %d element block(s) read against %d planned "
                      "element(s); %s went unweighed, so the section is written "
                      "again", index, len(rows), len(elements),
                      ", ".join(f"E{n}" for n in unweighed))
            yield "status", (f"Allegation {index} of {len(allegations)}: the "
                             f"element table was not fully read; writing it again")

        if element_note:
            notes.append(element_note)
        if notes:
            section = _note_before_disposition(section, notes)

        section, verdict, review = _settle_disposition(section, rows, elements,
                                                       index, labels)
        if element_note:
            # The decomposition itself was degraded - no elements at all, or an
            # element whose repetition marking could not be read - and the
            # dispositions table has to show that beside the verdict. Carried
            # only in the section's prose, an allegation weighed as one
            # undecomposed proposition printed as "1 of 1 element(s) weighed",
            # which is what a genuinely single-element allegation prints.
            note = "the element decomposition was incomplete or unreadable"
            review = f"{review}; {note}" if review else note

        if circular:
            log.warning("allegation %d: removed %d citation(s) to the allegation "
                        "itself", index, circular)
            section = (section.rstrip() +
                       f"\n\n*{circular} citation(s) to the allegation itself were "
                       f"removed: the allegation is the claim under test, not "
                       f"evidence.*\n")

        if not verdict:
            # _settle_disposition returns one of the two labels on every path,
            # so this is unreachable today. It is an error rather than a
            # fallback because the fallback would have been a label: a future
            # edit that let the disposition machinery return nothing would then
            # have printed an adverse disposition that no note, review string or
            # unsettled marker covered, and the harness would have counted it as
            # a weighed finding. A report is not written from a verdict nobody
            # produced.
            log.error("allegation %d: the disposition machinery produced no "
                      "label; the report is stopped rather than resolved toward "
                      "either one", index)
            yield "error", (f"Allegation {index} produced no disposition. This "
                            f"is a defect in the report generator, not a "
                            f"finding about the evidence; nothing has been "
                            f"saved.")
            return

        body.append(section.strip())
        dispositions.append({"index": index, "allegation": allegation,
                             "verdict": verdict,
                             "weighed": len(_weighed_elements(rows, elements)),
                             "planned": len(elements),
                             "citations": _grounded_citations(section,
                                                              document_names),
                             "review": review,
                             "findings": _findings_block(section)})

    yield "status", "Summary, persons, and timeline"
    # The summary is shown each allegation's findings as written, because it is
    # asked for the reason behind the disposition together with the citation
    # that reason rests on, and a pass given only the allegation and its verdict
    # has to invent the citation or scrape an unrelated one out of the timeline.
    dispo_lines = "\n\n".join(
        f"- Allegation {d['index']}: {d['allegation']}\n"
        f"  Disposition: {d['verdict']}\n"
        f"  Basis: {_basis(d)}\n"
        f"  Findings as written:\n" +
        "\n".join(f"    {line}" for line in d["findings"].splitlines())
        for d in dispositions)
    prompt = SUMMARY_TEMPLATE.format(corpus_note=banner.strip() or "(none)",
                                     dispositions=dispo_lines,
                                     timeline=_timeline_block(),
                                     entities=_entities_block(),
                                     yes_label=labels[0], no_label=labels[1])
    head = ""
    try:
        for token in client.stream(model, prompt, system=system, options=options,
                                   think=thinking_enabled()):
            head += token
            yield "token", token
    except Exception as exc:
        yield "error", str(exc).splitlines()[0]
        return

    # The summary is scrubbed like every other section. It is given the full
    # allegation text and told in prose not to cite it, and a circular citation
    # here survives into the one section every reader reads.
    head, head_circular = _scrub_allegation_citations(head, document_names)
    if head_circular:
        log.warning("summary: removed %d citation(s) to the allegation itself",
                    head_circular)
        head = (head.rstrip() +
                f"\n\n*{head_circular} citation(s) to the allegations themselves "
                f"were removed from this summary: an allegation is the claim "
                f"under test, not evidence.*\n")

    created = utcnow()
    contributed = evidence.contributing(contexts)
    done_count = sum(1 for d in docs if d.get("status") == "done")
    goal_block = f"## Goal\n\n{goal}\n\n" if goal else ""
    table = "\n".join(
        f"| {d['index']} | {d['allegation'][:90]} | **{d['verdict']}** "
        f"| {_basis(d)} |"
        for d in dispositions)
    dispo_table = ("## Dispositions\n\n"
                   "| # | Allegation | Disposition | Basis |\n|---|---|---|---|\n"
                   + table +
                   f"\n\n*(This table is assembled mechanically from the finding "
                   f"blocks below; it cannot disagree with them. Every "
                   f"disposition is one of: {', '.join(labels)}. The basis "
                   "column "
                   "is how much of each element table was weighed and how many "
                   "statements in the section cite a page of an ingested "
                   "document: a disposition resting on nothing says so here, "
                   "because the label alone cannot tell an allegation the "
                   "records refute from one the corpus never spoke to.)*\n\n")
    full = (f"# Report of Investigation\n\n"
            f"{banner}"
            f"Generated {created} from {len(docs)} ingested document(s), "
            f"{len(contributed)} of which contributed evidence, and "
            f"{facts_total['n']} extracted fact(s) using {model}.\n\n"
            f"{goal_block}"
            f"{dispo_table}"
            f"{head.strip()}\n\n## Findings by allegation\n\n"
            + "\n\n".join(body)
            + "\n\n---\n\nEvery statement above is drawn from the uploaded documents "
              "and cited to the page it came from. Machine transcription and "
              "extraction were used throughout; the page images remain the "
              "authority.\n")

    objective_record = (goal + "\n" + "\n".join(
        f"{i}. {a}" for i, a in enumerate(allegations, 1))).strip()
    report_id = hashlib.sha256(f"{created}|{objective_record}".encode()).hexdigest()[:16]
    with state.tx() as conn:
        conn.execute(
            """INSERT INTO reports (report_id, objective, body, model, documents,
                                    assertions, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (report_id, objective_record, full, model, done_count,
             facts_total["n"], created))
    log.info("report %s written (%d allegation(s), %d of %d document(s) "
             "contributed)", report_id, len(allegations), len(contributed), len(docs))
    yield "done", {"report_id": report_id}
