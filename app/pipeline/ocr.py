"""OCR pass, and the decision of whether a page needs the vision model.

Running a vision model over a clean typed page wastes minutes for a worse
result than Tesseract gives in a second.  Pages OCR reads confidently are
transcribed here; everything else goes to the VLM.  The bias is deliberate:
a wasted VLM pass costs minutes, a missed handwritten page costs the content.
"""
from __future__ import annotations

import pytesseract
from PIL import Image

from . import paths, state
from .config import env_int, env_str
from .ingest import write_text
from .log import get_logger

log = get_logger("ocr")

Image.MAX_IMAGE_PIXELS = 500_000_000
TESSERACT_CONFIG = "--oem 3 --psm 3"


def read_page(image_path, lang: str) -> tuple[str, float, int]:
    """Read a page, reconstructing lines by where words sit on the page.

    Tesseract's own block/paragraph/line numbering splits a wide table into one
    block per column, so grouping by it turns a row like

        1012   V-217   OUTBOUND   MSGT M. RUIZ

    into "1012 V-217" on one line and the rest somewhere else entirely - which
    silently destroys exactly the association a log table exists to record.
    Grouping by vertical position instead keeps rows intact, and the gaps
    between words are preserved as spacing so columns stay readable.
    """
    with Image.open(image_path) as im:
        data = pytesseract.image_to_data(im, lang=lang, config=TESSERACT_CONFIG,
                                         output_type=pytesseract.Output.DICT)

    words: list[dict] = []
    confs: list[float] = []
    for i, raw in enumerate(data["text"]):
        word = (raw or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not word or conf < 0:
            continue
        words.append({
            "text": word,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "height": max(1, int(data["height"][i])),
            "width": max(1, int(data["width"][i])),
        })
        confs.append(conf)

    if not words:
        return "", 0.0, 0

    # Two words belong to the same line when their vertical centres are within
    # roughly half a character height of each other.
    median_height = sorted(w["height"] for w in words)[len(words) // 2]
    tolerance = max(6, median_height * 0.6)

    rows: list[dict] = []
    for word in sorted(words, key=lambda w: w["top"] + w["height"] / 2):
        centre = word["top"] + word["height"] / 2
        if rows and abs(centre - rows[-1]["centre"]) <= tolerance:
            rows[-1]["words"].append(word)
            rows[-1]["centre"] = (rows[-1]["centre"] * (len(rows[-1]["words"]) - 1)
                                  + centre) / len(rows[-1]["words"])
        else:
            rows.append({"centre": centre, "words": [word]})

    # Within a line, order left to right and turn wide gaps into wide spacing so
    # column structure survives into the text the model reads.
    char_width = max(4, median_height * 0.5)
    lines_out: list[str] = []
    for row in rows:
        ordered = sorted(row["words"], key=lambda w: w["left"])
        line = ordered[0]["text"]
        cursor = ordered[0]["left"] + ordered[0]["width"]
        for word in ordered[1:]:
            gap = word["left"] - cursor
            # Always at least one space. Tesseract does sometimes split a word
            # into two boxes ("OUTBOUND" -> "OUT" "BOUND"), and it is tempting
            # to rejoin those by looking for a small gap - but measured on a
            # real page the split gap (0.24 of a character width) and a genuine
            # space in larger type (0.36) are too close to separate reliably.
            # Getting it wrong in the joining direction produces
            # "INSTALLATIONACCESSLOG", which corrupts search and embeddings;
            # getting it wrong the other way produces "OUT BOUND", which reads
            # fine and which quote checking already ignores whitespace for. So
            # the cheap, safe error is the one taken deliberately here.
            if gap < char_width:
                line += " " + word["text"]
            else:
                line += " " * min(8, int(gap / char_width)) + word["text"]
            cursor = word["left"] + word["width"]
        lines_out.append(line.rstrip())

    text = "\n".join(lines_out)
    mean_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
    return text, mean_conf, len(confs)


def run(doc_id: str, on_progress) -> dict:
    lang = env_str("TESSERACT_LANG", "eng")
    threshold = float(env_int("OCR_CONFIDENCE_THRESHOLD", 82))
    min_words = env_int("OCR_MIN_WORDS", 25)

    rows = state.query(
        """SELECT doc_id, page_num, image_path FROM pages
           WHERE doc_id=? AND image_path IS NOT NULL AND route IS NULL
           ORDER BY page_num""", (doc_id,))

    counts = {"ocr": 0, "vlm": 0}
    for idx, row in enumerate(rows, 1):
        on_progress(f"reading page {idx}/{len(rows)}")
        image = paths.under_root(row["image_path"])
        if image is None:
            continue

        text, conf, words = read_page(image, lang)
        clean_print = words >= min_words and conf >= threshold
        route = "ocr" if clean_print else "vlm"
        counts[route] += 1

        with state.tx() as conn:
            if clean_print:
                txt = write_text(doc_id, row["page_num"], text)
                conn.execute(
                    """UPDATE pages SET ocr_conf=?, ocr_words=?, route='ocr',
                           text_path=?, text_source='ocr', model='tesseract'
                       WHERE doc_id=? AND page_num=?""",
                    (conf, words, paths.rel(txt), doc_id, row["page_num"]))
            else:
                conn.execute(
                    """UPDATE pages SET ocr_conf=?, ocr_words=?, route='vlm'
                       WHERE doc_id=? AND page_num=?""",
                    (conf, words, doc_id, row["page_num"]))

    log.info("%s: %d page(s) read by OCR, %d routed to the vision model",
             doc_id, counts["ocr"], counts["vlm"])
    return counts
