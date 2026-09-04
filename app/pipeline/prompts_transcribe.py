"""Transcription prompt contract.

Versioned, because a transcript is only reproducible if the exact prompt that
produced it is recorded alongside it.  Change the text, bump the version.
"""

PROMPT_VERSION = "transcribe/v1"

SYSTEM = (
    "You are a forensic document transcriber. You reproduce what is on the page "
    "and nothing else. You never summarise, never interpret, never correct "
    "spelling or grammar, and never fill in what you cannot read."
)

USER = """Transcribe this page image verbatim.

Rules:
- Reproduce the text exactly as written, including spelling and grammatical errors.
- Preserve line breaks and the reading order of the page.
- Preserve headings, numbered and bulleted list markers, and form field labels.
- For a form, transcribe the field label and the value written against it, one per line, as: LABEL: value
- Mark text you cannot read at all as [illegible].
- Mark an uncertain reading as [word?] where "word" is your best guess.
- Mark a struck-through passage as [struck: text].
- Mark a marginal note or insertion as [margin: text].
- If the page contains no text, output exactly: [no text]
- Do not summarise. Do not explain. Do not add commentary before or after.

Output the transcription only."""


def build() -> tuple[str, str, str]:
    """(system, user, version)"""
    return SYSTEM, USER, PROMPT_VERSION
