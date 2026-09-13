"""Find text that is in a .docx but on no rendered page.

Conversion is the ingest path because only a layout engine produces pages.
What a renderer cannot be talked into printing is then not lost quietly:
every text unit in the file is checked against the pages that came out, and
anything absent is written to an annex page instead of disappearing.

The comparison is made on a stream of letters and digits with everything else
removed, which looks crude and is deliberate.  A PDF text extractor does not
reproduce the whitespace or the punctuation of the source.  Measured on real
files: pypdf breaks words in the middle of a line on justified text, so
'civilian cyber workforce shortage' came back as 'workforce sho\\nrtage', and
the reader writes a table row as 'cell | cell' where the page prints the cells
on separate lines.  Matching on words made both of those look like missing
text - 14 false positives across four documents.  Matching on letters and
digits alone reported 0, and a deliberate 15-word excision was still caught.

A failing unit is split at word boundaries and retried, so a paragraph that
crosses a page break fails whole and passes in halves while a genuine loss
fails all the way down.

Tracked deletions and hidden text are the one class that is not decided by
the comparison.  Strikethrough and the display setting are visual, and a PDF
text extractor keeps neither, so a struck-out or hidden sentence that the
renderer did print arrives as ordinary prose and can be quoted as if it had
stood.  When the pre-pass that labels those constructs did not run, they are
annexed with their label whatever the page shows.
"""
from __future__ import annotations

import re
import unicodedata
import zipfile
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
_PART = re.compile(
    r"^(word/(glossary/)?(document\d*|header\d+|footer\d+|footnotes"
    r"|endnotes|comments)\.xml"
    r"|word/(charts|diagrams|drawings|ink)/.*\.xml"
    r"|customXml/.*\.xml"
    r"|docProps/(core|app|custom)\.xml)$")
_KIND = {f"{W}t": "text", f"{W}delText": "deleted", f"{W}instrText": "field"}
_HIDE = (f"{W}vanish", f"{W}webHidden")
# Marked on the node itself rather than kept in a set of id()s: lxml hands out
# a fresh Python object for the same node once the old one has been freed, so
# an id()-keyed set reports the wrong runs as hidden.
HIDDEN = "{urn:docanalysis:docx}hidden"
FLOOR = 24            # normalised characters; shorter fragments are not split


def _norm(s: str) -> str:
    """Letters and digits only, accent-folded and case-folded.

    NFKD also decomposes the fi and fl ligatures a renderer emits, and drops
    every difference of spacing, hyphenation and punctuation between what the
    file stores and what an extractor returns.
    """
    return re.sub(r"[^0-9a-z]+", "",
                  unicodedata.normalize("NFKD", s).casefold())


def _mark_hidden(root) -> None:
    """Tag the text of every run Word is set not to display.

    The test is the one docx_prep removes on the copy it renders: a w:vanish
    or w:webHidden in the run's own properties, not switched off by w:val.  A
    vanish inherited from a character style is not followed here because
    docx_prep does not strip one either - such a run stays hidden in the
    render and is reported by the containment check like any other loss.
    """
    for tag in _HIDE:
        for prop in root.iter(tag):
            if prop.get(f"{W}val") not in (None, "true", "1", "on"):
                continue
            rpr = prop.getparent()
            run = rpr.getparent() if rpr is not None else None
            if run is None or run.tag != f"{W}r":
                continue
            for el in run.iter(f"{W}t", f"{W}delText"):
                el.set(HIDDEN, "1")


def units(src: Path) -> list[tuple[str, str, str]]:
    """Every run of text in the file as (part, kind, text).

    kind is 'text', 'deleted' (w:delText), 'hidden' (a run carrying w:vanish
    or w:webHidden) or 'field' (w:instrText).  Anything else carrying text in
    a content part is returned as 'text' too, so a construct nobody
    anticipated is checked rather than skipped.
    """
    out: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(src) as z:
        for name in sorted(z.namelist()):
            if not _PART.match(name):
                continue
            try:
                root = etree.fromstring(z.read(name))
            except etree.XMLSyntaxError:
                try:
                    root = etree.fromstring(z.read(name),
                                            etree.XMLParser(recover=True))
                except Exception:
                    continue
            # Modern Word writes a shape twice, DrawingML in mc:Choice and VML
            # in mc:Fallback.  Walking both doubles every text box, and a
            # doubled text box is a false fact in the corpus.
            for fb in list(root.iter(f"{MC}Fallback")):
                fb.getparent().remove(fb)
            _mark_hidden(root)
            for el in root.iter():
                if not isinstance(el.tag, str):
                    continue
                if el.text and el.text.strip():
                    kind = _KIND.get(el.tag, "text")
                    if kind == "text" and el.get(HIDDEN):
                        kind = "hidden"
                    out.append((name, kind, el.text))
    return out


def haystack(pages: list[str]) -> str:
    """The rendered pages as one normalised stream."""
    return _norm("\n".join(pages))


def fragments_missing(text: str, hay: str) -> list[str]:
    """Parts of text that appear nowhere in hay, split as far as FLOOR."""
    return _missing(text.split(), hay)


def _missing(words: list[str], hay: str) -> list[str]:
    n = _norm(" ".join(words))
    if not n or n in hay:
        return []
    if len(n) <= FLOOR or len(words) < 2:
        return [" ".join(words)]
    mid = len(words) // 2
    return _missing(words[:mid], hay) + _missing(words[mid:], hay)


def residue(src: Path, hay: str, *, report_deleted: bool
            ) -> list[tuple[str, str, str]]:
    """Fragments of src that appear on none of the rendered pages.

    Field instructions are never reported as loss: instruction text such as
    PAGEREF _Toc233281287 \\h is not printed by anything and its cached result
    is what reaches the page.  It is still carried to the annex by the caller.

    report_deleted is False when the pre-pass succeeded, because the deleted
    and hidden text was then unhidden and labelled and does render.  It is
    True when the pre-pass failed, so that deleted and hidden text is annexed
    whatever the render did - the rendered page may be carrying it with no
    mark on it.
    """
    out: list[tuple[str, str, str]] = []
    for part, kind, txt in units(src):
        if kind == "field":
            continue
        if kind in ("deleted", "hidden") and report_deleted:
            out.append((part, kind, txt))
            continue
        # docProps is the file's own metadata - template name, revision count,
        # word count, the author Word recorded. None of it is printed on a
        # page, so every value of it is "missing" by construction, and reported
        # here it would fill the annex with "Normal.dotm", "0" and "1" for
        # every document. The harvest carries docProps to the annex properly,
        # labelled as properties; this is the wrong place to say it twice.
        if part.startswith("docProps/"):
            continue
        # The floor belongs to the whole unit, not to the pieces it splits
        # into. A unit shorter than this - a bullet glyph, a cell holding "3",
        # a stray word - says nothing about the renderer whether it is found or
        # not, so it is not asked about. But once a unit IS long enough to ask
        # about, every piece of it that went missing is reported, however short
        # that piece is: a paragraph whose first sentence vanished has lost a
        # sentence, and the split reaching down to it is the point of the
        # split. Putting the floor on fragments instead silently discarded
        # exactly that case - measured, with a deleted paragraph heading going
        # unreported because it split down to seventeen characters.
        if len(_norm(txt)) < FLOOR:
            continue
        for frag in _missing(txt.split(), hay):
            out.append((part, kind, frag))
    return out
