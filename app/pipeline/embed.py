"""Passage embeddings, so the chat page can find the right pages to answer from.

Page text is split into overlapping passages, each embedded once and kept in
SQLite.  Search loads the vectors into one matrix and scores them with a dot
product - at the scale one machine's worth of documents reaches, that is
faster than maintaining an index and has nothing to get out of sync.
"""
from __future__ import annotations

import hashlib
import re
import threading

import numpy as np

from . import paths, state
from .config import env_str
from .log import get_logger, utcnow
from .model_client import Ollama


def _client() -> Ollama:
    """Embeddings get their own Ollama endpoint when one is configured.

    Two servers on two ports means the chat model and the embedding model stay
    resident side by side; sharing one server makes each request evict the
    other model and pay the reload every time.
    """
    # One Model Runner serves generation and embeddings alike, so this is the
    # same endpoint unless someone deliberately splits them.
    return Ollama(url=env_str("EMBED_MODEL_URL", "")
                  or env_str("EMBED_OLLAMA_URL", "")
                  or env_str("MODEL_URL", "")
                  or env_str("OLLAMA_URL", ""))

log = get_logger("embed")

TARGET_CHARS = 900      # passages large enough to carry an answer
OVERLAP_CHARS = 150     # so a fact split across a boundary is still findable
BATCH = 16

_cache_lock = threading.Lock()
_cache: dict | None = None


def chunk_text(text: str) -> list[str]:
    """Split on paragraphs, then sentences, keeping a little overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= TARGET_CHARS:
        return [text]

    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= TARGET_CHARS:
            pieces.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > TARGET_CHARS:
                pieces.append(current.strip())
                current = current[-OVERLAP_CHARS:] + " " + sentence
            else:
                current = f"{current} {sentence}".strip()
        if current.strip():
            pieces.append(current.strip())

    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) + 2 <= TARGET_CHARS:
            merged[-1] = f"{merged[-1]}\n{piece}"
        else:
            merged.append(piece)
    return merged


def chunk_id(doc_id: str, page_num: int, ord_: int, text: str) -> str:
    payload = f"{doc_id}|{page_num}|{ord_}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _invalidate() -> None:
    global _cache
    with _cache_lock:
        _cache = None


def run(doc_id: str, on_progress=lambda _m: None) -> int:
    """Embed every passage of a document that does not have a vector yet."""
    model = env_str("EMBED_MODEL", "").strip()
    if not model:
        log.info("EMBED_MODEL is not set; skipping embeddings (chat will be "
                 "limited to keyword search)")
        return 0

    pages = state.query(
        "SELECT page_num, text_path FROM pages WHERE doc_id=? AND text_path IS NOT NULL "
        "ORDER BY page_num", (doc_id,))

    pending: list[tuple] = []
    for page in pages:
        found = paths.under_root(page["text_path"])
        if found is None:
            continue
        for ord_, piece in enumerate(chunk_text(found.read_text(encoding="utf-8"))):
            cid = chunk_id(doc_id, page["page_num"], ord_, piece)
            existing = state.query_one(
                "SELECT embedding FROM chunks WHERE chunk_id=?", (cid,))
            if existing and existing["embedding"]:
                continue
            pending.append((cid, doc_id, page["page_num"], ord_, piece))

    if not pending:
        return 0

    client = _client()
    model = client.require_model(model, "EMBED_MODEL")
    done = 0
    for start in range(0, len(pending), BATCH):
        batch = pending[start:start + BATCH]
        on_progress(f"indexing passage {start + 1}-{start + len(batch)} of {len(pending)}")
        vectors = client.embed(model, [row[4] for row in batch])
        with state.tx() as conn:
            for row, vector in zip(batch, vectors):
                array = np.asarray(vector, dtype=np.float32)
                norm = float(np.linalg.norm(array))
                if norm:
                    array = array / norm      # stored normalised: search is a dot product
                conn.execute(
                    """INSERT INTO chunks (chunk_id, doc_id, page_num, ord, text,
                           embedding, dims, model, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                         embedding=excluded.embedding, dims=excluded.dims,
                         model=excluded.model""",
                    (row[0], row[1], row[2], row[3], row[4], array.tobytes(),
                     int(array.size), model, utcnow()))
                done += 1

    _invalidate()
    log.info("%s: indexed %d passage(s)", doc_id, done)
    return done


def _matrix() -> dict:
    """All vectors as one matrix, rebuilt when the chunk count changes."""
    global _cache
    with _cache_lock:
        count = state.query_one(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL")
        total = count["n"] if count else 0
        if _cache is not None and _cache["count"] == total:
            return _cache

        rows = state.query(
            """SELECT c.chunk_id, c.doc_id, c.page_num, c.text, c.embedding, c.dims,
                      d.filename
               FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
               WHERE c.embedding IS NOT NULL""")
        if not rows:
            _cache = {"count": 0, "matrix": None, "meta": []}
            return _cache

        dims = rows[0]["dims"]
        keep = [r for r in rows if r["dims"] == dims]
        if len(keep) != len(rows):
            # Changing EMBED_MODEL changes the vector width; mixing widths would
            # be meaningless, so the odd ones out are ignored until reindexed.
            log.warning("ignoring %d passage(s) embedded with a different model",
                        len(rows) - len(keep))

        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in keep])
        _cache = {
            "count": total,
            "matrix": matrix,
            "meta": [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
                      "page_num": r["page_num"], "text": r["text"],
                      "filename": r["filename"]} for r in keep],
        }
        return _cache


def search(question: str, k: int = 8) -> list[dict]:
    model = env_str("EMBED_MODEL", "").strip()
    cache = _matrix()
    if not model or cache["matrix"] is None:
        return keyword_search(question, k)

    client = _client()
    vector = np.asarray(client.embed(model, [question])[0], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    if vector.size != cache["matrix"].shape[1]:
        log.warning("question embedding is %d wide but the index is %d; "
                    "reindex after changing EMBED_MODEL",
                    vector.size, cache["matrix"].shape[1])
        return keyword_search(question, k)

    scores = cache["matrix"] @ vector
    top = np.argsort(-scores)[:k]
    return [{**cache["meta"][i], "score": round(float(scores[i]), 4)} for i in top]


def keyword_search(question: str, k: int = 8) -> list[dict]:
    """Fallback when there is no embedding model: match on the question's words."""
    words = [w for w in re.findall(r"\w{3,}", question.lower())][:12]
    if not words:
        return []
    clause = " OR ".join(["LOWER(c.text) LIKE ?"] * len(words))
    rows = state.query(
        f"""SELECT c.chunk_id, c.doc_id, c.page_num, c.text, d.filename
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE {clause} LIMIT 40""",
        [f"%{w}%" for w in words])

    scored = []
    for row in rows:
        text = row["text"].lower()
        hits = sum(1 for w in words if w in text)
        scored.append({"chunk_id": row["chunk_id"], "doc_id": row["doc_id"],
                       "page_num": row["page_num"], "text": row["text"],
                       "filename": row["filename"], "score": hits})
    scored.sort(key=lambda r: -r["score"])
    return scored[:k]


def backfill(on_progress=lambda _m: None) -> int:
    """Index everything already processed - used after setting EMBED_MODEL."""
    total = 0
    for row in state.query("SELECT doc_id FROM documents ORDER BY uploaded_at"):
        total += run(row["doc_id"], on_progress)
    return total


def stats() -> dict:
    row = state.query_one(
        "SELECT COUNT(*) AS n, SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) "
        "AS embedded FROM chunks")
    return {"chunks": row["n"] if row else 0,
            "embedded": (row["embedded"] or 0) if row else 0}
