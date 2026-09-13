"""Read a .docx by walking the package XML itself.

python-docx reaches body paragraphs and body tables. Everything else a Word
file can hold - headers, footers, footnotes, endnotes, comments, text boxes,
content controls, field results, tracked changes, SmartArt labels, embedded
workbooks - is text the file contains and python-docx never returns.

Three things this module does that a reader cannot do by asking python-docx:

  Pages.  Word records its own pagination in the file as w:lastRenderedPageBreak,
  written at every place its layout engine broke a page at the last save. Those
  marks, plus authored breaks, reconstruct Word's page numbers without laying
  the document out again. docProps/app.xml <Pages> is Word's own page count, so
  the reconstruction can be checked against it and the document marked when the
  two disagree.

  Everything else.  Every text-bearing part is walked, and then every part is
  swept a second time for text this module's tag list did not claim. The residue
  is emitted verbatim rather than dropped, so a construct nobody anticipated
  arrives as unlabelled text instead of silence.

  Pictures.  Embedded images are written out as page images and left for the
  existing OCR and vision routes, because a scan pasted into a Word file is
  the one kind of text that is in no XML at all.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
PIC = "{http://schemas.openxmlformats.org/drawingml/2006/picture}"

# Consumption is marked on the element itself, never by id(). lxml hands out a
# fresh Python object for the same node whenever the old one has been freed, so
# a set of id()s silently reports elements as read that were never read - and
# the whole point of the sweep at the end is that it cannot be fooled that way.
READ = "{urn:docanalysis:docx}read"
READ_ATTRS = "{urn:docanalysis:docx}read-attrs"


def _claim(el, attr: str | None = None) -> None:
    if attr is None:
        el.set(READ, "1")
        return
    have = el.get(READ_ATTRS) or ""
    el.set(READ_ATTRS, f"{have} {attr}".strip())


def _claimed(el, attr: str | None = None) -> bool:
    if attr is None:
        return el.get(READ) is not None
    return attr in (el.get(READ_ATTRS) or "").split()


_BODY_PART = re.compile(r"^word/document\d*\.xml$")
_TEXT_PART = re.compile(
    r"^(?:word|word/glossary)/"
    r"(?:document\d*|header\d+|footer\d+|footnotes|endnotes|comments|"
    r"commentsExtended|people)\.xml$"
    r"|^word/(?:charts|diagrams|drawings|ink|embeddings)/.*\.xml$"
    r"|^customXml/item\d*\.xml$"
    r"|^docProps/(?:core|custom|app)\.xml$")
# Attributes that carry prose. Text is not always element text: a picture's
# alternative text, a field's instruction and a form field's stored value are
# all attributes, and all of them are words somebody wrote.
_TEXT_ATTRS = (W + "instr", "descr", "title", W + "alias", W + "tag")
_MEDIA = re.compile(r"^word/media/.+\.(?:png|jpe?g|tiff?|bmp|gif|emf|wmf)$", re.I)


@dataclass
class DocxRead:
    pages: list[str] = field(default_factory=list)
    annex: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)      # zip names
    basis: str = "single"           # word | explicit | estimated | single
    word_pages: int | None = None   # what Word itself recorded
    warnings: list[str] = field(default_factory=list)


class _Sink:
    def __init__(self) -> None:
        self.pages: list[list[str]] = []
        self.lines: list[str] = []
        self.buf: list[str] = []
        self.dirty = False
        self.requests = 0           # break requests, including suppressed ones
        self.notes: list[list[tuple[str, str]]] = [[]]   # (kind, id) per page

    def add(self, text: str) -> None:
        if text:
            self.buf.append(text)
            self.dirty = True

    def flush(self) -> None:
        line = "".join(self.buf).strip()
        self.buf.clear()
        if line:
            self.lines.append(line)

    def note(self, kind: str, ident: str) -> None:
        self.notes[-1].append((kind, ident))

    def brk(self) -> None:
        """A page ends here.

        Counted even when suppressed: a break at the very start of a table cell
        is the normal shape of a row that straddles a page, and the cell's own
        sink has no text before it to end.
        """
        self.requests += 1
        self.flush()
        if not self.dirty:
            return                  # nothing since the last break: not a page
        self.pages.append(self.lines)
        self.lines = []
        self.notes.append([])
        self.dirty = False

    def done(self) -> list[list[str]]:
        self.flush()
        self.pages.append(self.lines)
        self.lines = []
        return self.pages


def _is_hidden(el) -> bool:
    run = el.getparent()
    if run is None or run.tag != W + "r":
        return False
    rpr = run.find(W + "rPr")
    if rpr is None:
        return False
    vanish = rpr.find(W + "vanish")
    return vanish is not None and (vanish.get(W + "val") or "true") not in ("0", "false")


def _walk(el, sink: _Sink, seen: set[int]) -> None:
    tag = el.tag
    if not isinstance(tag, str):
        return
    if tag == W + "p":
        ppr = el.find(W + "pPr")
        if ppr is not None:
            pbb = ppr.find(W + "pageBreakBefore")
            if pbb is not None and (pbb.get(W + "val") or "true") not in ("0", "false"):
                sink.brk()
        for child in el:
            _walk(child, sink, seen)
        sink.flush()
        return
    if tag == W + "tbl":
        _table(el, sink, seen)
        return
    if tag in (W + "t", M + "t", A + "t"):
        _claim(el)
        text = el.text or ""
        if tag == W + "t" and _is_hidden(el):
            # Hidden text: in the file, never on the page. A renderer cannot
            # print it and so cannot capture it; here it is kept and marked,
            # because text somebody hid is not text an investigation drops.
            text = f"[hidden: {text}]"
        sink.add(text)
        if tag == A + "t":
            sink.flush()
        return
    if tag == W + "delText":
        _claim(el)
        # Tracked deletion. Word does not display it and a renderer cannot
        # print it; what a draft used to say is evidence, so it is marked and
        # kept rather than dropped.
        sink.add(f"[deleted: {el.text or ''}]")
        return
    if tag == W + "instrText":
        _claim(el)
        if (el.text or "").strip():
            sink.add(f" [field: {el.text.strip()}] ")
        return
    if tag == W + "lastRenderedPageBreak":
        sink.brk()
        return
    if tag == W + "br":
        sink.brk() if el.get(W + "type") == "page" else sink.flush()
        return
    if tag == W + "cr":
        sink.flush()
        return
    if tag == W + "tab":
        sink.add("\t")
        return
    if tag == W + "noBreakHyphen":
        sink.add("-")
        return
    if tag in (W + "footnoteReference", W + "endnoteReference", W + "commentReference"):
        _claim(el)
        kind = {"footnoteReference": "footnote", "endnoteReference": "endnote",
                "commentReference": "comment"}[etree.QName(el).localname]
        ident = el.get(W + "id") or "?"
        sink.add(f" [{kind} {ident}] ")
        sink.note(kind, ident)
        return
    if tag == W + "fldSimple":
        _claim(el, W + "instr")
        instr = (el.get(W + "instr") or "").strip()
        if instr:
            sink.add(f" [field: {instr}] ")
        for child in el:
            _walk(child, sink, seen)
        return
    for child in el:
        _walk(child, sink, seen)


def _table(tbl, sink: _Sink, seen: set[int]) -> None:
    for row in tbl.findall(W + "tr"):
        cells: list[str] = []
        rowbreak = False
        for cell in row.findall(W + "tc"):
            sub = _Sink()
            for child in cell:
                _walk(child, sub, seen)
            sub.flush()
            if sub.requests:
                rowbreak = True
            cells.append(" / ".join([l for pg in sub.pages for l in pg] + sub.lines))
        if any(c.strip() for c in cells):
            sink.lines.append(" | ".join(cells))
            sink.dirty = True
        if rowbreak:
            # A row that straddles a page. The break lands at the row boundary
            # rather than inside the row: the page number stays right, the
            # split is one row coarse.
            sink.brk()


def _strip_fallbacks(root, stripped: list) -> None:
    """mc:Fallback repeats mc:Choice for older Word versions - the same words
    twice. Harvesting both duplicates every modern text box, so the fallback
    is taken out of the walk. It is kept, not discarded: the sweep at the end
    puts back anything the fallback said that its mc:Choice did not, because
    'a duplicate' is an assumption and this module does not drop text on one."""
    for fb in list(root.iter(MC + "Fallback")):
        parent = fb.getparent()
        if parent is not None:
            parent.remove(fb)
            stripped.append(fb)


def _residue(root, seen: set[int], part: str) -> list[str]:
    """Text this module's tag list did not claim.

    The guarantee is not that the list above is complete - it is that anything
    missing from it still comes out. Every element is revisited; any that holds
    text and was never consumed is emitted with its tag, so an unknown
    construct arrives as labelled text rather than as silence.
    """
    out: list[str] = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        name = etree.QName(el).localname
        text = (el.text or "").strip()
        if text and id(el) not in seen and name not in _IGNORE_TEXT:
            out.append(f"[{part} {name}] {text}")
        for attr in _TEXT_ATTRS:
            val = (el.get(attr) or "").strip()
            if val and id(el) not in seen:
                out.append(f"[{part} {name}@{etree.QName(attr).localname}] {val}")
    return out


# Elements whose text is markup bookkeeping, not prose.
_IGNORE_TEXT = {"Relationship", "Default", "Override"}

_NOTE_PART = {"footnotes": W + "footnote", "endnotes": W + "endnote",
              "comments": W + "comment"}


def _notes(root, part: str, seen: set) -> dict[str, list[str]]:
    """id -> lines, for footnotes.xml / endnotes.xml / comments.xml."""
    if root is None:
        return {}
    out: dict[str, list[str]] = {}
    container = _NOTE_PART[part]
    for node in root:
        if node.tag != container:
            continue
        if (node.get(W + "type") or "") in ("separator", "continuationSeparator"):
            continue
        sink = _Sink()
        for child in node:
            _walk(child, sink, seen)
        lines = [l for pg in sink.done() for l in pg]
        if not lines:
            continue
        label = ""
        author, date = node.get(W + "author"), node.get(W + "date")
        if author:
            label = f" by {author}"
        if date:
            label += f" {date}"
        out[node.get(W + "id") or "?"] = [f"[{part[:-1]} {node.get(W + 'id')}{label}]"] + lines
    return out


def _part_lines(root, seen: set[int]) -> list[str]:
    sink = _Sink()
    _walk(root, sink, seen)
    return [l for pg in sink.done() for l in pg]


def _external_links(zf: zipfile.ZipFile) -> list[str]:
    """A URL a document points at is evidence, and it is in no w:t: hyperlink
    targets live in the .rels beside the part, not in the text."""
    seen: list[str] = []
    for name in zf.namelist():
        if not name.endswith(".rels"):
            continue
        try:
            root = etree.fromstring(zf.read(name))
        except etree.XMLSyntaxError:
            continue
        for rel in root.iter(REL + "Relationship"):
            if rel.get("TargetMode") == "External":
                target = (rel.get("Target") or "").strip()
                if target and target not in seen:
                    seen.append(target)
    return seen


def _embedded(zf: zipfile.ZipFile) -> list[str]:
    """Objects embedded in the document: a pasted workbook, a nested Word file.

    An OOXML embedding is itself a zip, so one level down is read. A legacy
    OLE .bin is not readable here and is named rather than ignored, because an
    object nobody can see is worse than one that announces itself.
    """
    import io
    out: list[str] = []
    for name in zf.namelist():
        if not name.startswith("word/embeddings/"):
            continue
        blob = zf.read(name)
        if blob[:2] != b"PK":
            out.append(f"[embedded object {name}, {len(blob)} bytes, "
                       f"not readable as XML - open the original to read it]")
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as inner:
                texts: list[str] = []
                for sub in inner.namelist():
                    if not sub.endswith(".xml"):
                        continue
                    try:
                        root = etree.fromstring(inner.read(sub))
                    except etree.XMLSyntaxError:
                        continue
                    for el in root.iter():
                        if isinstance(el.tag, str) and (el.text or "").strip():
                            texts.append(el.text.strip())
                if texts:
                    out.append(f"[embedded object {name}]")
                    out.extend(texts)
        except zipfile.BadZipFile:
            out.append(f"[embedded object {name}, {len(blob)} bytes, unreadable]")
    return out


_VAL_TAGS = {"docVar", "listEntry", "default"}


def _sweep_residue(roots: dict, stripped: list, seen: set,
                   emitted: str) -> list[str]:
    """Text in the package that the walk above did not claim.

    The promise is not that the tag list in _walk is complete - it is that
    anything missing from it still comes out. Every part is revisited and every
    element holding text that was never consumed is emitted with the part and
    tag it came from, so a construct this module has never heard of arrives as
    labelled text instead of as silence.
    """
    out: list[str] = []
    for name in sorted(roots):
        root = roots[name]
        if root is None:
            out.append(f"[{name} is not readable XML and could not be searched]")
            continue
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            local = etree.QName(el).localname
            text = (el.text or "").strip()
            if text and not _claimed(el) and local not in _IGNORE_TEXT:
                out.append(f"[unclaimed {name} {local}] {text}")
            for attr in _TEXT_ATTRS:
                val = (el.get(attr) or "").strip()
                if val and not _claimed(el, attr):
                    out.append(f"[{name} {local}@{etree.QName(attr).localname}] {val}")
            if local in _VAL_TAGS:
                for attr in (W + "name", W + "val"):
                    val = (el.get(attr) or "").strip()
                    if val:
                        out.append(f"[{name} {local}@{etree.QName(attr).localname}] {val}")
    seen_values: set = set()
    out = [line for line in out
           if not (line in seen_values or seen_values.add(line))]
    for fb in stripped:
        for el in fb.iter():
            if not isinstance(el.tag, str):
                continue
            text = (el.text or "").strip()
            if text and text not in emitted:
                out.append(f"[compatibility fallback {etree.QName(el).localname}] {text}")
    return out


def _estimate(lines: list[str], budget: int) -> list[list[str]]:
    """No layout evidence in the file: cut at paragraph boundaries on a
    character budget. These are not Word's pages and are labelled as such
    everywhere they surface."""
    pages: list[list[str]] = [[]]
    size = 0
    for line in lines:
        if size and size + len(line) > budget:
            pages.append([])
            size = 0
        pages[-1].append(line)
        size += len(line) + 2
    return pages


def read_docx(src: str | Path, *, estimate_budget: int = 2800) -> DocxRead:
    """Read the file. Every part is parsed once, up front, and the trees are
    held for the whole call: the sweep at the end identifies consumed elements
    by identity, and a tree that had been freed would let a later part reuse
    its addresses and look read when it never was."""
    out = DocxRead()
    seen: set = set()
    roots: dict = {}
    stripped: list = []
    with zipfile.ZipFile(str(src)) as zf:
        names = zf.namelist()
        for name in names:
            if not name.endswith(".xml") or name.endswith(".rels"):
                continue
            try:
                roots[name] = etree.fromstring(zf.read(name))
            except etree.XMLSyntaxError as exc:
                # A truncated or damaged part. Half a document is not nothing,
                # and a reader that returns nothing here hands the operator a
                # failed upload with no idea what the file held - so what can
                # be parsed is parsed, and the damage is stated on the page
                # rather than left to be discovered by its absence.
                try:
                    roots[name] = etree.fromstring(
                        zf.read(name), etree.XMLParser(recover=True))
                    out.warnings.append(
                        f"{name} inside this file is damaged ({exc}). What "
                        "could still be read from it is below; text after the "
                        "damage is NOT, and is not in this record at all.")
                except (etree.XMLSyntaxError, KeyError, RuntimeError):
                    roots[name] = None
                    out.warnings.append(
                        f"{name} inside this file could not be read at all "
                        f"({exc}). Its text is not in this record.")
            except (KeyError, RuntimeError):
                roots[name] = None
        body_name = next((n for n in ("word/document.xml", "word/document2.xml")
                          if roots.get(n) is not None), None)
        if body_name is None:
            raise RuntimeError(
                "this .docx has no word/document.xml that can be read")
        for root in roots.values():
            if root is not None:
                _strip_fallbacks(root, stripped)

        notes = {kind: _notes(roots.get(f"word/{kind}.xml"), kind, seen)
                 for kind in ("footnotes", "endnotes", "comments")}

        body_root = roots[body_name]
        sink = _Sink()
        _walk(body_root, sink, seen)
        body_pages = sink.done()
        per_page_notes = sink.notes
        out.word_pages = _word_page_count(roots.get("docProps/app.xml"))

        if any(True for _ in body_root.iter(W + "lastRenderedPageBreak")):
            out.basis = "word"
        elif len(body_pages) > 1:
            out.basis = "explicit"
        else:
            flat = [l for pg in body_pages for l in pg]
            total = sum(len(l) for l in flat)
            if total > estimate_budget:
                # Word itself never laid this file out - it was written by a
                # library, or saved by something that does not paginate. The
                # cut is this reader's, and it says so wherever it surfaces.
                budget = estimate_budget
                if out.word_pages and out.word_pages > 1:
                    budget = max(500, total // out.word_pages)
                body_pages = _estimate(flat, budget)
                per_page_notes = [[] for _ in body_pages]
                per_page_notes[0] = [n for page in sink.notes for n in page]
                out.basis = "estimated"
                out.warnings.append(
                    "This file records no pagination of its own, so these page "
                    "numbers are this reader's estimate and will not match the "
                    "page numbers a reader sees in Word.")
            else:
                out.basis = "single"

        # A footnote prints at the foot of the page its reference sits on and
        # a comment points at the text it was left on, so both are attached to
        # that page rather than banished to the end where no citation reaches.
        for index, page_lines in enumerate(body_pages):
            attached: list[str] = []
            for kind, ident in (per_page_notes[index]
                                if index < len(per_page_notes) else []):
                if kind == "endnote":
                    continue
                block = notes.get(kind + "s", {}).get(ident)
                if block:
                    attached.extend(block)
            out.pages.append("\n\n".join(page_lines + attached))

        for name in sorted(roots):
            root = roots[name]
            if root is None or name == body_name:
                continue
            if re.match(r"^word/(?:header|footer)\d+\.xml$", name):
                lines = _part_lines(root, seen)
                if lines:
                    out.annex.append(
                        f"[{name} - printed on every page of its section]\n"
                        + "\n\n".join(lines))
            elif (re.match(r"^word/glossary/.+\.xml$", name)
                  or re.match(r"^word/(?:charts|diagrams|drawings|ink)/.+\.xml$", name)
                  or re.match(r"^customXml/item\d*\.xml$", name)):
                lines = _part_lines(root, seen)
                if lines:
                    out.annex.append(f"[{name}]\n" + "\n\n".join(lines))

        endnotes = notes.get("endnotes", {})
        if endnotes:
            out.annex.append("[endnotes - printed at the end of the document]\n"
                             + "\n\n".join("\n".join(b) for b in endnotes.values()))
        out.annex.extend(_properties(roots))
        links = _external_links(zf)
        if links:
            out.annex.append("[hyperlink targets]\n" + "\n".join(links))
        embedded = _embedded(zf)
        if embedded:
            out.annex.append("\n".join(embedded))
        out.images = [n for n in names if _MEDIA.match(n)]
        if out.images:
            out.annex.append("[pictures in this document, read separately as "
                             "page images]\n" + "\n".join(out.images))

        emitted = "\n".join(out.pages) + "\n" + "\n".join(out.annex)
        residue = _sweep_residue(roots, stripped, seen, emitted)
        if residue:
            out.annex.append(
                "[text in this file that this reader did not recognise, kept "
                "verbatim so that nothing in the file is lost]\n"
                + "\n".join(residue))

    if out.word_pages and out.basis in ("word", "explicit") \
            and out.word_pages != len(out.pages):
        out.warnings.append(
            f"Word recorded {out.word_pages} pages in this file and this reader "
            f"reconstructed {len(out.pages)}. Page numbers after the first "
            "disagreement are shifted by that difference; treat them as "
            "approximate.")
    return out


def _word_page_count(root) -> int | None:
    """Word's own page count, written into docProps/app.xml at the last save."""
    if root is None:
        return None
    for el in root:
        if isinstance(el.tag, str) and etree.QName(el).localname == "Pages":
            try:
                return int((el.text or "").strip())
            except ValueError:
                return None
    return None


_PROP_LABELS = {
    "title": "title", "subject": "subject", "description": "description",
    "creator": "author", "lastModifiedBy": "last saved by",
    "keywords": "keywords", "category": "category", "revision": "revision",
    "created": "created", "modified": "modified", "lastPrinted": "last printed",
    "Company": "company", "Manager": "manager", "Application": "written by",
    "Template": "template", "TotalTime": "minutes spent editing",
    "Pages": "pages Word counted", "Words": "words Word counted",
    "LastPrinted": "last printed",
}


def _properties(roots: dict) -> list[str]:
    """Who wrote it, when, with what, and what Word counted.

    Named and labelled rather than left to the sweep, because a file's
    properties are the part of it a reader is most likely to want quoted and
    the least likely to look at: an author, a company, a template, an editing
    time. Everything in these parts is claimed here, including the counters,
    so that what the sweep reports later is genuinely unexpected.
    """
    named: list[str] = []
    stats: list[str] = []
    for part in ("docProps/core.xml", "docProps/app.xml"):
        root = roots.get(part)
        if root is None:
            continue
        for el in root.iter():
            if not isinstance(el.tag, str) or el is root:
                continue
            local = etree.QName(el).localname
            text = (el.text or "").strip()
            _claim(el)
            if not text:
                continue
            label = _PROP_LABELS.get(local)
            if label:
                named.append(f"{label}: {text}")
            else:
                stats.append(f"{local}={text}")
    root = roots.get("docProps/custom.xml")
    if root is not None:
        for prop in root:
            if not isinstance(prop.tag, str):
                continue
            name = prop.get("name") or "custom property"
            for child in prop:
                _claim(child)
                if (child.text or "").strip():
                    named.append(f"{name}: {child.text.strip()}")
            _claim(prop)
    out = []
    if named:
        out.append("[document properties]\n" + "\n".join(named))
    if stats:
        out.append("[file statistics Word kept]\n" + " ".join(stats))
    return out


def iter_images(src: str | Path):
    """The pictures embedded in the file, as (name, bytes).

    A photograph of a page pasted into a Word document is the one kind of text
    that is in none of the XML, and this pipeline already knows how to read a
    picture of a page. So the pictures come out and take the route every other
    page image takes - OCR, and the vision model where OCR is not clean.
    """
    with zipfile.ZipFile(str(src)) as zf:
        for name in zf.namelist():
            if _MEDIA.match(name):
                yield name, zf.read(name)
