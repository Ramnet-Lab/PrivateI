"""Runs the stages for an uploaded document, one document at a time.

A single worker thread, deliberately: transcription and extraction both hit the
same GPU through Ollama, and running two documents at once makes both slower
rather than either faster.
"""
from __future__ import annotations

import queue
import threading
import traceback

from . import embed, extract, graph, ingest, ocr, state, transcribe
from .log import get_logger
from .model_client import ModelMissing, ModelNotSet, OllamaError

log = get_logger("runner")

_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_lock = threading.Lock()
# What is genuinely queued or running right now. The guard against double
# processing must NOT be inferred from the status column: upload writes
# status='queued' before calling enqueue(), so a guard reading that column
# refuses every fresh upload - marked queued, never actually queued.
_pending: set[str] = set()
_pending_lock = threading.Lock()


def enqueue(doc_id: str) -> None:
    with _pending_lock:
        if doc_id in _pending:
            log.info("%s is already queued or running; not adding it again", doc_id)
            return
        _pending.add(doc_id)
    state.set_status(doc_id, "queued", stage="queued", progress="waiting to start")
    _queue.put(doc_id)
    _ensure_worker()


def queue_depth() -> int:
    return _queue.qsize()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="pipeline", daemon=True)
            _worker.start()


def _loop() -> None:
    while True:
        doc_id = _queue.get()
        try:
            process(doc_id)
        except Exception:
            log.error("unhandled failure processing %s\n%s", doc_id, traceback.format_exc())
            state.set_status(doc_id, "failed", stage="failed",
                             error="unexpected error; see the app log")
        finally:
            with _pending_lock:
                _pending.discard(doc_id)
            _queue.task_done()


def process(doc_id: str) -> None:
    # Deleted while waiting its turn: nothing to do, and every later stage
    # would crash on the missing row.
    if not state.query_one("SELECT 1 AS n FROM documents WHERE doc_id=?", (doc_id,)):
        log.info("%s was deleted before processing started; skipping", doc_id)
        return

    def progress(message: str, stage: str | None = None) -> None:
        state.set_status(doc_id, "processing", stage=stage, progress=message)

    notes: list[str] = []
    progress("starting", stage="reading")

    try:
        ingest.run(doc_id, lambda m: progress(m, "reading"))
    except Exception as exc:
        log.error("%s: ingest failed: %s", doc_id, exc)
        state.set_status(doc_id, "failed", stage="reading", error=str(exc)[:400])
        return

    try:
        ocr.run(doc_id, lambda m: progress(m, "ocr"))
    except Exception as exc:
        notes.append(f"OCR problem: {exc}")
        log.error("%s: ocr failed: %s", doc_id, exc)

    # Vision model: needed only for pages OCR could not read.
    try:
        transcribe.run(doc_id, lambda m: progress(m, "transcribing"))
    except (ModelNotSet, ModelMissing) as exc:
        pending = state.query_one(
            "SELECT COUNT(*) AS n FROM pages WHERE doc_id=? AND route='vlm' "
            "AND text_path IS NULL", (doc_id,))
        if pending and pending["n"]:
            notes.append(f"{pending['n']} page(s) need a vision model. "
                         f"{str(exc).splitlines()[0]}")
    except OllamaError as exc:
        notes.append(str(exc).splitlines()[0])

    read = state.query_one(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN text_path IS NOT NULL THEN 1 ELSE 0 END) AS with_text
           FROM pages WHERE doc_id=?""", (doc_id,))
    total_pages = read["total"] if read else 0
    pages_with_text = (read["with_text"] or 0) if read else 0

    if total_pages and pages_with_text == 0:
        # Nothing was readable, so there is nothing to extract from.  Saying
        # "done" here would claim the document had no content, which is a
        # different statement from "nothing could be read".
        state.set_status(
            doc_id, "incomplete", stage="reading",
            progress=f"no text could be read from {total_pages} page(s)",
            error="; ".join(notes)[:400] or
                  "these pages need a vision model - set VLM_MODEL in .env")
        return
    if pages_with_text < total_pages:
        notes.append(f"{total_pages - pages_with_text} of {total_pages} page(s) "
                     f"could not be read")

    # Passage index for the chat page.  Done before extraction so questions
    # can be asked as soon as the text exists, model for facts or not.
    progress("indexing passages for chat", "indexing")
    try:
        embed.run(doc_id, lambda m: progress(m, "indexing"))
    except Exception as exc:
        notes.append(f"passage indexing problem: {str(exc).splitlines()[0]}")
        log.error("%s: embedding failed: %s", doc_id, exc)

    # Text model: without it there are no assertions, so no graph.
    try:
        kept, dropped = extract.run(doc_id, lambda m: progress(m, "extracting"))
        if dropped:
            notes.append(f"{dropped} assertion(s) discarded: quote not on the page")
    except (ModelNotSet, ModelMissing, OllamaError) as exc:
        state.set_status(doc_id, "text_only", stage="extracting",
                         progress="text extracted; no graph built",
                         error=str(exc).splitlines()[0])
        log.warning("%s: no text model, stopping after transcription", doc_id)
        return

    progress("resolving duplicate names", "linking")
    try:
        extract.auto_merge(lambda m: progress(m, "linking"))
    except Exception as exc:
        notes.append(f"name resolution problem: {exc}")

    progress("building the graph", "linking")
    try:
        graph.load(doc_id, lambda m: progress(m, "linking"))
    except Exception as exc:
        state.set_status(doc_id, "failed", stage="linking",
                         error=f"could not reach the graph database: {exc}"[:400])
        return

    total = state.query_one(
        "SELECT COUNT(*) AS n FROM triples WHERE doc_id=?", (doc_id,))
    state.set_status(doc_id, "done", stage="done",
                     progress=f"{total['n'] if total else 0} assertion(s) in the graph",
                     error="; ".join(notes)[:400] or None)
    log.info("%s: finished", doc_id)


def requeue_unfinished() -> int:
    """After a restart, pick up anything that was mid-flight."""
    rows = state.query(
        "SELECT doc_id FROM documents WHERE status IN ('queued','processing')")
    for row in rows:
        with _pending_lock:
            if row["doc_id"] in _pending:
                continue
            _pending.add(row["doc_id"])
        _queue.put(row["doc_id"])
    if rows:
        _ensure_worker()
    return len(rows)
