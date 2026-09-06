"""Runs the stages for an uploaded document, one document at a time.

A single worker thread, deliberately: transcription and extraction both hit the
same GPU through Ollama, and running two documents at once makes both slower
rather than either faster.

That one thread spends most of its life blocked inside a single model HTTP
call, which is why superseding a run takes more than another queue entry. An
operator who changes the model endpoint and presses re-ingest would otherwise
wait out the remainder of a call that is still using the settings they just
replaced, watching a button that appears to do nothing. So every run carries a
cancellation token, and the routes that supersede a run cancel that token
before they queue its replacement.
"""
from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
import traceback

from . import embed, extract, graph, ingest, ocr, state, transcribe
from .log import get_logger
from neo4j.exceptions import AuthError, ServiceUnavailable

# The cancellation half of this lives in model_client, because the socket a
# blocked call is waiting on is only reachable from there. Three names are the
# whole of the contract: CancelToken() is a handle whose cancel() may be called
# from any thread and whose .cancelled says whether it has been,
# cancellation_scope(token) binds a token to this thread so the requests made
# on it answer to that token, and Cancelled is what an abandoned request
# raises. Cancelled is a BaseException there, on purpose, so that the broad
# "except Exception" a stage wraps each page in cannot mistake an abandonment
# for one more unreadable page.
from .model_client import (Cancelled, CancelToken, ModelMissing, ModelNotSet,
                           OllamaError, cancellation_scope)

log = get_logger("runner")

_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_lock = threading.Lock()
# What is genuinely queued or running right now. The guard against double
# processing must NOT be inferred from the status column: upload writes
# status='queued' before calling enqueue(), so a guard reading that column
# refuses every fresh upload - marked queued, never actually queued.
#
# A count rather than a set, because superseding a run puts a document in the
# queue while a run of it is still unwinding: for that moment the document
# genuinely has two claims on it, and a set cannot tell the run that is ending
# whether the entry it is about to drop is its own or its replacement's.
_pending: dict[str, int] = {}
_pending_lock = threading.Lock()

# How long a caller that has just superseded a run waits for the worker to
# actually let go of it. The wait is what lets that caller delete the
# document's rows knowing the abandoned run has stopped writing to them. On
# expiry it carries on anyway - the worker may be inside a stage that makes no
# model calls at all - which is exactly what every caller did before any of
# this existed.
CANCEL_GRACE_SECONDS = 5.0


class _Run:
    """One execution of process(), and the token that can abandon it.

    The token is made here and never reused, because cancelling is permanent: a
    token shared between two runs would carry the first run's abort into the
    second. That is also what keeps a late cancel harmless. A caller cancels
    the token it captured rather than "whatever the worker is doing now", so a
    cancel that arrives after its run has ended finds a dead token and stops
    there - it cannot reach the run that replaced it, which is holding a token
    that caller never saw.
    """

    __slots__ = ("doc_id", "generation", "token", "done", "superseded")

    def __init__(self, doc_id: str, generation: int) -> None:
        self.doc_id = doc_id
        self.generation = generation
        # The document names the token, so that a call abandoned three modules
        # away still says in its message which document it belonged to.
        self.token = CancelToken(doc_id)
        # Set by the caller that cancels, before it cancels: it says that this
        # run is being replaced rather than merely stopped, and it is set first
        # so that the worker cannot see the abort without also seeing why.
        self.superseded = False
        # Set when the run has finished unwinding and settled its row, so a
        # caller that cancelled can wait for the document to be its own again.
        self.done = threading.Event()


# The run the worker is inside right now, and the number of the last one
# started. The generation is for the log: it is what makes "run 41 abandoned"
# and "starting run 42" legible as one supersession rather than two events.
_current: _Run | None = None
_current_lock = threading.Lock()

# Held open in normal running. A re-ingest closes it while it cancels, resets
# and re-queues, because those three steps are one change and the worker must
# not act on a half-made one: freeing the worker by cancelling, then deleting
# rows, lets it start the next queued document and write pages into a document
# this request is in the middle of wiping.
_gate = threading.Event()
_gate.set()


@contextmanager
def paused():
    """Hold the worker at the start of its next document until this returns."""
    _gate.clear()
    try:
        yield
    finally:
        _gate.set()
_generation = 0


def enqueue(doc_id: str, force: bool = False) -> None:
    """Add a document to the work queue.

    force=True is for re-ingest, which has already deleted the document's
    pages and assertions. Without it the guard could refuse the re-add while
    the reset had already thrown the old results away, leaving the document
    wiped and unqueued - a wedge that no retry could clear because the stale
    claim never expired.
    """
    with _pending_lock:
        outstanding = _pending.get(doc_id, 0)
        if outstanding:
            if not force:
                log.info("%s is already queued or running; not adding it again",
                         doc_id)
                return
            log.info("%s: re-ingest supersedes the in-flight run", doc_id)
        _pending[doc_id] = outstanding + 1
        # Written while the count is held because a run being abandoned settles
        # the same row under the same lock, and the two orders have to agree.
        # Outside the lock they can interleave into a document marked failed
        # that is in fact queued, which would then sit there saying so for as
        # long as the queue in front of it.
        state.set_status(doc_id, "queued", stage="queued",
                         progress="waiting to start")
    _queue.put(doc_id)
    _ensure_worker()


def cancel_inflight(doc_id: str | None = None,
                    grace: float = CANCEL_GRACE_SECONDS) -> str | None:
    """Abandon the run the worker is inside right now; return its document.

    doc_id=None means whatever is running, which is what re-ingesting the whole
    corpus wants: it supersedes every run there can be. Naming a document
    cancels only a run of that document, because re-ingesting one document does
    not supersede another document's run.

    Callers must queue the replacement for whatever this abandons, and must do
    that AFTER calling here. The cancel is aimed at the token this call
    captured, never at the worker thread, which is what keeps a cancel meant
    for run N from reaching run N+1: by the time the worker starts N+1 it holds
    a different token and this one is spent. Cancelling before queueing is the
    other half of the same rule - a cancel issued once the replacement is
    queued could find that the worker had already started it.

    Returns None when nothing was in flight, in which case nothing has
    happened at all and the caller behaves as it always did.
    """
    with _current_lock:
        run = _current
        if run is None or (doc_id is not None and run.doc_id != doc_id):
            return None
    # Outside the lock: this reaches into a request another thread is blocked
    # on, and that thread takes this same lock to retire its run.
    log.info("%s: aborting run %d and the model call it is inside",
             run.doc_id, run.generation)
    run.superseded = True
    run.token.cancel()
    if grace > 0 and not run.done.wait(grace):
        log.warning("%s: run %d has not let go after %.0fs; continuing without "
                    "it", run.doc_id, run.generation, grace)
    return run.doc_id


def inflight() -> str | None:
    """The document the worker is inside right now, or None.

    A caller that has just cancelled needs to know whether the run actually
    let go. Deletion in particular cannot treat a timed-out cancel as a
    completed one: the abandoned run is still writing.
    """
    with _current_lock:
        return _current.doc_id if _current is not None else None


def queue_depth() -> int:
    return _queue.qsize()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="pipeline", daemon=True)
            _worker.start()


def _begin(doc_id: str) -> _Run:
    """Publish the run about to start, so it can be cancelled while it runs."""
    global _current, _generation
    with _current_lock:
        _generation += 1
        _current = _Run(doc_id, _generation)
        return _current


def _finish(run: _Run, abandoned: bool) -> None:
    """Retire a run: stop it being cancellable, and settle its document's row.

    A document abandoned on purpose must not be left saying "processing" with
    nobody processing it. Which honest thing to say depends on whether anything
    is replacing this run, and that is decided while the claim count is held so
    that it cannot change between the reading of it and the writing.
    """
    global _current
    with _current_lock:
        if _current is run:
            _current = None
    with _pending_lock:
        outstanding = _pending.get(run.doc_id, 1) - 1
        if outstanding > 0:
            _pending[run.doc_id] = outstanding
        else:
            _pending.pop(run.doc_id, None)
        if abandoned:
            if run.superseded or outstanding > 0:
                # Either the replacement is already queued, or the caller that
                # cancelled is on its way to queueing it - it waits for this
                # run to let go before it does, so the two orders both arrive
                # here. Queued is what the document is in both, and it cannot
                # have started again yet: one worker, and it is here. Saying
                # anything else, even for the second it would take the caller
                # to correct it, puts "failed" against a document the operator
                # has just asked for.
                state.set_status(run.doc_id, "queued", stage="queued",
                                 progress="superseded; waiting to start again")
            else:
                # Abandoned by something that never said it was replacing
                # this run. Nothing is coming for the document, so failed is
                # the honest word for a half-read document, and both the retry
                # and re-ingest buttons clear it.
                state.set_status(
                    run.doc_id, "failed", stage="cancelled",
                    error="the run was cancelled and nothing replaced it; "
                          "press Re-ingest to read this document again")
    run.done.set()


def _loop() -> None:
    while True:
        doc_id = _queue.get()
        # Before anything is read or written for this document. A re-ingest
        # that has closed the gate is mid-way through resetting rows, and
        # starting here would race it.
        _gate.wait()
        run = _begin(doc_id)
        abandoned = False
        try:
            # Binding the token to this thread is what makes the model calls
            # the stages issue abortable; the stages themselves build their own
            # clients and know nothing about any of this.
            with cancellation_scope(run.token):
                process(doc_id, run.token)
        except Cancelled:
            # Not a failure. This document was abandoned on purpose by someone
            # who has queued the run that replaces it, so it is logged as the
            # supersession it is and its row is settled in _finish rather than
            # being stamped with an error the operator would have to interpret.
            abandoned = True
            log.info("%s: run %d abandoned; the work is being started again",
                     doc_id, run.generation)
        except Exception:
            log.error("unhandled failure processing %s\n%s", doc_id, traceback.format_exc())
            state.set_status(doc_id, "failed", stage="failed",
                             error="unexpected error; see the app log")
        finally:
            _finish(run, abandoned)
            _queue.task_done()


def process(doc_id: str, token: CancelToken | None = None) -> None:
    # Deleted while waiting its turn: nothing to do, and every later stage
    # would crash on the missing row.
    if not state.query_one("SELECT 1 AS n FROM documents WHERE doc_id=?", (doc_id,)):
        log.info("%s was deleted before processing started; skipping", doc_id)
        return

    def progress(message: str, stage: str | None = None) -> None:
        # Every stage reports between pages or between batches, so this is
        # where an abandoned run notices it was superseded while it was NOT
        # inside a model call - between two of them, or somewhere like OCR that
        # makes none at all and would otherwise run to the end of the document
        # before anything looked. It reaches inside loops this module does not
        # own, because they all report progress. The check comes before the
        # write, or an abandoned run's last act is to claim it is still
        # processing.
        if token is not None and token.cancelled:
            raise Cancelled(f"{doc_id}: superseded by a newer run")
        state.set_status(doc_id, "processing", stage=stage, progress=message)

    notes: list[str] = []
    progress("starting", stage="reading")

    # Every stage below is wrapped in a broad handler, because one stage's
    # failure is not the whole document's. An abandonment is not a failure and
    # none of them may absorb it, so each says so in its own right rather than
    # relying on Cancelled being a BaseException over in model_client: the day
    # somebody makes that class an ordinary Exception - the obvious-looking
    # tidy-up - these handlers would otherwise start marking abandoned
    # documents text_only and done, which is the one outcome this must never
    # produce.
    try:
        ingest.run(doc_id, lambda m: progress(m, "reading"))
    except Cancelled:
        raise
    except Exception as exc:
        log.error("%s: ingest failed: %s", doc_id, exc)
        state.set_status(doc_id, "failed", stage="reading", error=str(exc)[:400])
        return

    try:
        ocr.run(doc_id, lambda m: progress(m, "ocr"))
    except Cancelled:
        raise
    except Exception as exc:
        notes.append(f"OCR problem: {exc}")
        log.error("%s: ocr failed: %s", doc_id, exc)

    # Vision model: needed only for pages OCR could not read.
    try:
        transcribe.run(doc_id, lambda m: progress(m, "transcribing"))
    except Cancelled:
        raise
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
    except Cancelled:
        raise
    except Exception as exc:
        notes.append(f"passage indexing problem: {str(exc).splitlines()[0]}")
        log.error("%s: embedding failed: %s", doc_id, exc)

    # Text model: without it there are no assertions, so no graph.
    try:
        kept, dropped = extract.run(doc_id, lambda m: progress(m, "extracting"))
        if dropped:
            notes.append(f"{dropped} assertion(s) discarded: quote not on the page")
    except Cancelled:
        raise
    except (ModelNotSet, ModelMissing, OllamaError) as exc:
        state.set_status(doc_id, "text_only", stage="extracting",
                         progress="text extracted; no graph built",
                         error=str(exc).splitlines()[0])
        log.warning("%s: no text model, stopping after transcription", doc_id)
        return

    # Extraction reports its progress a page at a time, so an abort during the
    # last page of the last document is the one that can reach here with the
    # run already abandoned. Everything below writes a terminal status, and a
    # document abandoned part way through extraction that ends up marked done
    # is exactly the half-written result this must never produce.
    if token is not None and token.cancelled:
        raise Cancelled(f"{doc_id}: superseded by a newer run")

    progress("resolving duplicate names", "linking")
    try:
        extract.auto_merge(lambda m: progress(m, "linking"))
    except Cancelled:
        raise
    except Exception as exc:
        notes.append(f"name resolution problem: {exc}")

    progress("building the graph", "linking")
    try:
        graph.load(doc_id, lambda m: progress(m, "linking"))
    except Cancelled:
        raise
    except (ServiceUnavailable, AuthError) as exc:
        state.set_status(doc_id, "failed", stage="linking",
                         error=f"could not reach the graph database: {exc}"[:400])
        return
    except Exception as exc:
        # Anything else is a fault in our own loading code. Saying "could not
        # reach the graph database" for an AttributeError sends the reader to
        # check Neo4j while the real bug sits in this repo.
        log.exception("%s: graph load failed", doc_id)
        state.set_status(doc_id, "failed", stage="linking",
                         error=f"graph load failed: {type(exc).__name__}: {exc}"[:400])
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
            if _pending.get(row["doc_id"]):
                continue
            _pending[row["doc_id"]] = 1
        _queue.put(row["doc_id"])
    if rows:
        _ensure_worker()
    return len(rows)
