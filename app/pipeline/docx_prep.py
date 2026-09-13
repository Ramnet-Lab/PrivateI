"""Rewrite a copy of a .docx so the renderer prints what Word hides.

A renderer prints the DISPLAY view.  A Word file can be set to display
neither its tracked deletions nor its hidden text, and LibreOffice obeys that
setting exactly as Word does - measured, a probe carrying
<w:revisionView w:markup="false"/> rendered 'before  after' where the deleted
sentence had been.  Text that is in the file must reach the corpus, so the
suppression is removed from a COPY before rendering.  The original file is
never touched: it is the evidence, and its sha256 is the dedupe key.

Deleted and hidden text are labelled as they are unhidden.  Strikethrough is
a visual attribute and every PDF text extractor discards it, so without a
label a struck-out sentence enters the corpus as ordinary prose and can be
quoted as if it had stood.  If this pass fails, the caller renders the
original and force-annexes every deleted and hidden unit instead, so that
text is never both present and unmarked.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

from lxml import etree

from .log import get_logger

log = get_logger("docx_prep")

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

# Every part whose contents a reader would expect to see on the page, plus
# settings.xml, which is where the display suppression itself lives.  Header
# and footer are prefixes: a file carries header1.xml, header2.xml and so on,
# one per section.  Parts not named here - styles.xml above all - are left
# alone deliberately: text hidden by a style rather than by a direct run
# property cannot be labelled from where the suppression sits, and text that
# never renders is caught by the audit pass and annexed with its label there.
_PARTS = ("word/settings.xml", "word/document.xml", "word/header",
          "word/footer", "word/footnotes.xml", "word/endnotes.xml",
          "word/glossary/document.xml")

# Values Word writes for an on/off property when it means "on".  An explicit
# w:val="false" or "0" is a run switching the property back OFF - usually a
# run inside a hidden paragraph that was meant to stay visible - and removing
# that would hide text rather than reveal it.
_ON = (None, "true", "1", "on")


def _label(el, opening: str, closing: str = "]") -> None:
    """Bracket the text this element carries, in document order.

    One opening mark on the first text node and one closing mark on the last,
    rather than a pair per run: a deleted sentence is usually several runs and
    per-run bracketing would come out as noise around every word of it.
    """
    texts = [t for t in el.iter(f"{W}t", f"{W}delText") if t.text]
    if not texts:
        return
    texts[0].text = opening + texts[0].text
    texts[-1].text = texts[-1].text + closing
    # The label adds a space next to the text, and without xml:space="preserve"
    # Word and LibreOffice are both entitled to collapse it away.
    for t in texts:
        t.set(XML_SPACE, "preserve")


def _rewrite(root, mark: bool) -> int:
    """Unhide every suppressed construct in one part; return how many."""
    changed = 0

    # settings.xml: the file-wide instruction not to display revisions at all.
    # It has to go first in intent, though not in order - while it stands,
    # unwrapping a deletion below would still render nothing.
    for el in list(root.iter(f"{W}revisionView")):
        el.getparent().remove(el)
        changed += 1

    # A tracked deletion is a w:del wrapper around ordinary runs whose text
    # sits in w:delText instead of w:t.  Unwrapping the runs into the parent
    # and renaming the text nodes turns it back into text that renders; the
    # label is applied first, while the delText nodes are still there to find.
    for dele in list(root.iter(f"{W}del")):
        parent = dele.getparent()
        if parent is None:
            continue
        if mark:
            _label(dele, "[struck: ")
        for dt in dele.iter(f"{W}delText"):
            dt.tag = f"{W}t"
        at = list(parent).index(dele)
        for child in reversed(list(dele)):
            parent.insert(at, child)
        parent.remove(dele)
        changed += 1

    # Hidden text is a run property, not a wrapper: the run is ordinary and
    # carries w:vanish (hidden in print and on screen) or w:webHidden (hidden
    # in web layout).  Removing the property is all it takes.  The same
    # property on a paragraph mark's rPr hides the paragraph mark itself, and
    # that is worth removing too - there is no run to label in that case, so
    # the label is only applied when the property's grandparent really is a run.
    for tag in (f"{W}vanish", f"{W}webHidden"):
        for prop in list(root.iter(tag)):
            if prop.get(f"{W}val") not in _ON:
                continue
            rpr = prop.getparent()
            if rpr is None:
                continue
            run = rpr.getparent()
            rpr.remove(prop)
            if mark and run is not None and run.tag == f"{W}r":
                _label(run, "[hidden: ")
            changed += 1
    return changed


def prepare(src: Path, dest: Path, *, mark: bool = True) -> int:
    """Write a renderable copy of src to dest; return constructs unhidden.

    src is opened read-only and never written to.  dest is built beside
    itself and moved into place, so a failure part-way through leaves no
    half-written .docx for the renderer to find: the caller catches the
    exception, renders the ORIGINAL, and records that the page may carry
    deleted or hidden text with no mark on it.
    """
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    changed = 0
    try:
        with zipfile.ZipFile(src) as zin, \
                zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if any(item.filename.startswith(p) for p in _PARTS):
                    try:
                        root = etree.fromstring(data)
                    except etree.XMLSyntaxError as exc:
                        # One unparseable part is not a reason to lose the
                        # other nineteen.  It goes across as it came in, and
                        # anything it was hiding is the audit pass's problem.
                        log.warning("%s: %s did not parse (%s); copied as is",
                                    src.name, item.filename, exc)
                        zout.writestr(item, data)
                        continue
                    n = _rewrite(root, mark)
                    if n:
                        changed += n
                        data = etree.tostring(root, xml_declaration=True,
                                              encoding="UTF-8", standalone=True)
                zout.writestr(item, data)
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return changed
