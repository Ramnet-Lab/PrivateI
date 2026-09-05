"""Extraction prompt contract.  Versioned like the transcription prompt."""

PROMPT_VERSION = "extract/v11"

ENTITY_TYPES = ["PERSON", "ORG", "LOCATION", "EVENT", "DOCUMENT", "CLAIM"]

SYSTEM = (
    "You are an analyst extracting structured assertions from an investigative "
    "document. You only record what the text states. You never infer, never "
    "combine facts to reach a conclusion, and never use outside knowledge. "
    "Every assertion you output must be supported by a quote copied verbatim "
    "from the text you were given."
)

USER_TEMPLATE = """Extract every factual assertion from the page below as subject-predicate-object triples.

Entity types: {types}

Rules:
- subject_type and object_type must be one of the entity types listed above.
- predicate is a short lowercase verb phrase, e.g. "attended", "reported to", "was located at", "signed", "stated".
- ABSENCE IS EVIDENCE. A statement that something did NOT happen, was NOT done, or was NOT observed is a fact and must be extracted with the same care as a positive one. In an investigation these are often the most probative items there are: a check that was not performed, a seal that was not replaced, an email that was never answered, a witness who did not see the thing everyone assumes they saw. Do not skip a sentence because it reports a non-event.
- OBSERVATION LIMITS ARE FACTS. When a person states a limit on what they could perceive - too far away, view blocked, wearing earbuds, arrived after it started, only heard part of it - extract that as an assertion about them. It is what lets a reader weigh their account against someone with a clearer view.
- NEGATION IS SACRED. "never replied" must yield predicate "never replied to" - not predicate "replied" with the negation buried in the object, and never a double form like "replied never replied". Flipping a negative statement positive is the worst possible error here: it survives a skim and reverses a finding. The same for "did not", "refused to", "failed to", "denied".
- BOUNDARIES ARE NOT DATES. "logged zero checks after <a date>" is a statement about the period AFTER that date - the date itself is a day the checks WERE logged. Keep after/before/since/until inside the assertion text ("logged zero weekly checks after <that date>") and leave event_date null; event_date is only for something that happened ON a date.
- A predicate must never end in "since", "until", "as of", or "prior to". Those words take a time, and the object is not one - a predicate ending in one of them produces nonsense like "was in the section since <a thing that is not a time>". Keep the boundary word inside the object text and leave event_date null.
- Use the name exactly as the text writes it. Do not expand, normalise, or resolve abbreviations - "<rank> <surname>" keeps its rank.
- STATED NICKNAMES ARE FACTS. When the text says a person is called something else - "<rank> <surname>, who everyone calls <nickname>", "goes by <nickname>", "we called him <nickname>" - emit an assertion with predicate "is also known as": subject the full name, object_type PERSON, object the nickname. A nickname cannot be inferred from spelling, so the only way it is ever known is that a document said it.
- A JOB TITLE IS NOT A PERSON. "Civilian Network Administrator", "the section chief", "Equipment Test Craftsman", "the flight chief" are roles. If the text names the person holding the role, use the name; if it does not, skip the assertion rather than making the title into a person. Roles may be used as the OBJECT of a role assertion ("PERSON: <the named person>" / "holds the position of" / "CLAIM: <the title>").
- NEVER invent a name. Every subject_name and object_name must appear verbatim somewhere in this page or in the document header below. If you cannot name a person from the text, skip the assertion entirely.
- Pronoun resolution in interview transcripts: "I", "me", "my" refer to the interviewee named in the document header. "You" inside an interviewer's question also refers to that interviewee. Resolve them to that person's name; never emit "I", "you", "the interviewee", or "unknown" as an entity name, and never substitute some other person's name.
- A CLAIM's text is written in the third person. Replace "I", "me", "my", "myself" inside the claim with the name of the person the claim is attributed to - the subject of your own assertion - and "you" in an interviewer's question with the interviewee's name. Leave "we" and "us" as written; a group cannot be resolved to one person. A first-person word left standing inside a claim is read later as a second person present at the event, which turns two accounts that agree into a contradiction.
- event_date_basis: how the date is known - "stated" (the text gives the exact day), "month" (only a month is given: emit "YYYY-MM", basis "month"), "approx" ("around mid May": leave event_date null OR give the month with basis "approx"), "inferred" (you resolved a relative expression from an explicit anchor). Omit or null when event_date is null.
- event_date: WHEN THE EVENT ITSELF HAPPENED, not when it was written down, reported, or investigated. This distinction matters more than any other field here.
  * A statement written on one date, describing something that happened on an earlier date, takes the EARLIER date - the day of the event, not the day of the writing.
  * "During an interview on <date B>, <a person> said the remark was made on <date A>" -> the remark has event_date <date A>. Use <date B> only if the assertion is literally about the interview taking place.
  * Include the time whenever the text gives one: YYYY-MM-DDTHH:MM. Times matter as much as dates in an investigation.
  * Convert 24-hour times: "0925" -> T09:25, "1540" -> T15:40, "at 1015" -> T10:15.
  * Resolve relative wording against the date established in the passage: "the same day", "that afternoon", "later that morning" take the date they refer back to.
  * If the text gives a time but no date anywhere, leave event_date null rather than inventing a date.
  * "since <month> <year>" or "in <month>" names a MONTH, not a day. Emit it as "YYYY-MM" with event_date_basis "month" - never basis "stated", which claims the document gave a date it did not, and never a day the text did not give.
  * Never write a clock time the text did not give. If the page states no time, event_date ends at the day - do not pad it to T00:00, which claims a midnight nobody testified to.
  * Preserve the text's own precision. "Around mid May" or "early June" is NOT a date - leave event_date null and let the quote carry the wording. Never sharpen an approximation into a specific day.
  * Resolve a relative expression ("the following Monday", "two days later") ONLY when the passage states the anchor date explicitly; otherwise null.
  * Never guess a year. Use null when the page does not state when it happened.
  * ONE ASSERTION PER DATE. When a sentence attaches the same predicate to SEVERAL dates - "four advances, dated <d1>, <d2>, <d3> and <d4>", "late on <d1>, <d2> and <d3>", "checks logged <d1> and <d2>" - emit one assertion for EACH date, not a single summary carrying none of them. A ledger line, a log extract or a schedule is a list of separate dated events that happen to share a sentence. One field cannot hold four dates, so a summary silently discards every one of them and the chronology loses the whole series. Distribute ONLY the dates the text attributes to that subject: if a person is said to have missed one of the occasions, or to have been at only one, do not hand them the others. Splitting a sentence must never manufacture an attendance the sentence denies.
  * A DATE INSIDE YOUR OWN QUOTE IS A DATE YOU HAVE. If the quote you are citing names when the thing happened - "your written complaint of <date>", "the memo of <date>", "the <date> email" - that assertion has an event_date. Leaving it null while the date sits in the quote throws away something the document stated plainly.
- quote: a span copied character for character from the page text that supports the triple. It must appear in the page text exactly. Do not paraphrase.
- A statement someone made is a CLAIM: subject is the person, predicate "stated", object_type "CLAIM", and object_name the claim itself. Do not repeat the type inside the name - write "he attended the 0900 staff meeting", not "CLAIM: he attended the 0900 staff meeting".
- When a line introduces a numbered allegation ("Allegation 2." on its own or after a speaker tag), everything the speaker says after it until the next such line is testimony about THAT allegation. Do not carry a statement across such a line or blend one allegation's answer into another's.
- Skip boilerplate, letterheads, page numbers, and form instructions.
- Kin and possessives resolve to the speaker: "my dad" from the interviewee's mouth becomes "<interviewee's name>'s father" - never a bare "my dad" and never first person inside a name.
- Ignore [illegible] spans. Do not invent content for them.
- If the page supports no assertions, return an empty list.

Return JSON only, in exactly this shape:
{{"triples": [
  {{"subject_type": "PERSON", "subject_name": "SSgt Smith",
    "predicate": "attended",
    "object_type": "EVENT", "object_name": "0900 staff meeting",
    "event_date": "2026-03-14T09:00", "event_date_basis": "stated",
    "quote": "SSgt Smith was present at the 0900 staff meeting"}}
]}}

DOCUMENT HEADER (page 1 of this document - identifies the interviewee/author;
use it to resolve first- and second-person pronouns):
---
{header}
---

PAGE TEXT ({doc_id} page {page_num}):
---
{text}
---"""


def build(doc_id: str, page_num: int, text: str,
          header: str = "") -> tuple[str, str, str]:
    user = USER_TEMPLATE.format(types=", ".join(ENTITY_TYPES), doc_id=doc_id,
                                page_num=page_num, text=text,
                                header=header.strip() or "(none)")
    return SYSTEM, user, PROMPT_VERSION
