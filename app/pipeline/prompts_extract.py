"""Extraction prompt contract.  Versioned like the transcription prompt."""

PROMPT_VERSION = "extract/v5"

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
- NEGATION IS SACRED. "never replied" must yield predicate "never replied to" - not predicate "replied" with the negation buried in the object, and never a double form like "replied never replied". Flipping a negative statement positive is the worst possible error here: it survives a skim and reverses a finding. The same for "did not", "refused to", "failed to", "denied".
- BOUNDARIES ARE NOT DATES. "logged zero checks after 11 May" is a statement about the period AFTER that date - 11 May itself is a day the checks WERE logged. Keep after/before/since/until inside the assertion text ("logged zero weekly checks after 2026-05-11") and leave event_date null; event_date is only for something that happened ON a date.
- Use the name exactly as the text writes it. Do not expand, normalise, or resolve abbreviations - "SSgt Smith" stays "SSgt Smith".
- A JOB TITLE IS NOT A PERSON. "Civilian Network Administrator", "the section chief", "Equipment Test Craftsman", "the flight chief" are roles. If the text names the person holding the role, use the name; if it does not, skip the assertion rather than making the title into a person. Roles may be used as the OBJECT of a role assertion ("PERSON: Ana Reyes" / "holds the position of" / "CLAIM: Equipment Test Craftsman").
- NEVER invent a name. Every subject_name and object_name must appear verbatim somewhere in this page or in the document header below. If you cannot name a person from the text, skip the assertion entirely.
- Pronoun resolution in interview transcripts: "I", "me", "my" refer to the interviewee named in the document header. "You" inside an interviewer's question also refers to that interviewee. Resolve them to that person's name; never emit "I", "you", "the interviewee", or "unknown" as an entity name, and never substitute some other person's name.
- event_date_basis: how the date is known - "stated" (the text gives the exact day), "month" (only a month is given: use the month, day 01, basis "month"), "approx" ("around mid May": leave event_date null OR give the month with basis "approx"), "inferred" (you resolved a relative expression from an explicit anchor). Omit or null when event_date is null.
- event_date: WHEN THE EVENT ITSELF HAPPENED, not when it was written down, reported, or investigated. This distinction matters more than any other field here.
  * A statement dated 8 June describing something that happened on 14 May has event_date 2026-05-14, not the statement's own date.
  * "During an interview on 10 June, Brandt said the remark was made on 22 May" -> the remark has event_date 2026-05-22. Use 10 June only if the assertion is literally about the interview taking place.
  * Include the time whenever the text gives one: YYYY-MM-DDTHH:MM. Times matter as much as dates in an investigation.
  * Convert 24-hour times: "0925" -> T09:25, "1540" -> T15:40, "at 1015" -> T10:15.
  * Resolve relative wording against the date established in the passage: "the same day", "that afternoon", "later that morning" take the date they refer back to.
  * If the text gives a time but no date anywhere, leave event_date null rather than inventing a date.
  * Preserve the text's own precision. "Around mid May" or "early June" is NOT a date - leave event_date null and let the quote carry the wording. Never sharpen an approximation into a specific day.
  * Resolve a relative expression ("the following Monday", "two days later") ONLY when the passage states the anchor date explicitly; otherwise null.
  * Never guess a year. Use null when the page does not state when it happened.
- quote: a span copied character for character from the page text that supports the triple. It must appear in the page text exactly. Do not paraphrase.
- A statement someone made is a CLAIM: subject is the person, predicate "stated", object_type "CLAIM", and object_name the claim itself. Do not repeat the type inside the name - write "he attended the 0900 staff meeting", not "CLAIM: he attended the 0900 staff meeting".
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
