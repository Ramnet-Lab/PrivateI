"""Canonical data tree layout.  One tree, so deleting a case is one directory."""
from __future__ import annotations

import os
import re
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))

RAW = DATA_ROOT / "01_raw"
PAGES = DATA_ROOT / "02_pages"
TEXT = DATA_ROOT / "03_text"
PRODUCTS = DATA_ROOT / "05_products"
LOGS = DATA_ROOT / "99_logs"
STATE_DB = DATA_ROOT / "state.db"

SUPPORTED_PDF = {".pdf"}
SUPPORTED_DOC = {".docx"}
SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".heic", ".webp"}
SUPPORTED = SUPPORTED_PDF | SUPPORTED_DOC | SUPPORTED_IMAGE

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def ensure_tree() -> None:
    for d in (RAW, PAGES, TEXT, PRODUCTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """Uploads name files; a name is never trusted as a path."""
    cleaned = _SAFE.sub("_", Path(name).name).strip("._") or "upload"
    return cleaned[:120]


def doc_id_for(filename: str, sha256: str) -> str:
    stem = _SAFE.sub("_", Path(filename).stem).strip("_")[:60] or "doc"
    return f"{stem}__{sha256[:8]}"


def page_image(doc_id: str, page_num: int) -> Path:
    return PAGES / doc_id / f"page_{page_num:04d}.png"


def model_image(doc_id: str, page_num: int) -> Path:
    """The reduced copy of a page that is sent to a vision model.

    Beside the page image rather than in a temporary directory, and deliberately
    kept: when a transcript is wrong the first question is what the model was
    actually shown, and on this pipeline that has already been the answer once.
    It lives inside the document's own folder, so deleting or re-ingesting the
    document takes it with everything else.
    """
    return PAGES / doc_id / f"page_{page_num:04d}.model.jpg"


def transcript_txt(doc_id: str, page_num: int) -> Path:
    return TEXT / doc_id / f"page_{page_num:04d}.txt"


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(DATA_ROOT))
    except ValueError:
        return str(p)


def under_root(relative: str) -> Path | None:
    """Resolve a stored relative path, refusing anything that escapes the tree."""
    candidate = (DATA_ROOT / relative).resolve()
    if not str(candidate).startswith(str(DATA_ROOT.resolve())):
        return None
    return candidate if candidate.exists() else None
