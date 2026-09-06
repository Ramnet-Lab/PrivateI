"""One report being written, outliving the request that asked for it.

Writing a report is many minutes of local model work.  Run inside the request
that asked for it, it belongs to the browser: a refresh, a closed tab or a shut
laptop closes the response, the generator is torn down at whatever yield it was
sitting on, and an hour of machine time is gone.  Nothing survives it either,
because the reports row is written only once the last pass has finished.

So the run happens here, on a thread of its own, and the request only watches.
Watching can stop and start as often as the operator likes; the work does not
notice.  What a watcher needs is everything emitted so far and then everything
emitted from now on, which is what the entry buffer and follow() are.

This module knows nothing about reports.  It is handed a producer - anything
that yields (kind, data) pairs - so that what a report *is* stays in one place
and this stays a thing that runs a producer and remembers what it said.

One run at a time, for the same reason the document worker is one thread: the
passes hit the same GPU, and two reports at once make both slower rather than
either faster.  A second request is refused rather than queued, because each
run is a full report and an operator who clicked twice did not ask for two.

Nothing here is persisted.  A run dies with the process, and there is no
half-written report to resume from - unlike a document, whose durable truth is
its row and which runner.requeue_unfinished() picks up again at startup.  After
a restart there is simply no active run, and the last completed report is in the
reports table where it always was.
"""
from __future__ import annotations

import secrets
import threading
from typing import Any, Callable, Iterable, Iterator

from .log import get_logger, utcnow
from .model_client import Cancelled, CancelToken, cancellation_scope

log = get_logger("report_run")

# How long a watcher parks before sending a keepalive.  Some proxy or browser
# between here and the page will close a connection that has said nothing for
# long enough, and a report can spend minutes inside one model call without a
# word.  The comment frame costs nothing and is dropped by the client.
KEEPALIVE_SECONDS = 15.0

# A ceiling on remembered text.  The buffer is what a reattaching page is
# replayed from, so it holds the whole report rather than a tail - but a
# producer that never stops writing must not take the process with it.  Past
# this the text is dropped and said to be dropped, which is worse than the
# report and better than the alternative.
MAX_TEXT_CHARS = 4 * 1024 * 1024

RUNNING = "running"


class Busy(RuntimeError):
    """A run is already in flight.  Carries the run_id of the one that is."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"a report is already being written ({run_id})")
        self.run_id = run_id


class Run:
    """One execution of a producer, everything it said, and its abort switch.

    Entries are append-only and never edited once written.  That is what makes
    a plain integer a sufficient cursor: a watcher that has seen entry 40 has
    seen all of entry 40, for ever, and can be resumed from 41 by anyone.
    Coalescing streamed tokens into the entry before them would save a list
    slot and cost that guarantee - a watcher already past an entry would never
    learn that it had grown.  They are merged when they are sent instead, which
    is where the saving was wanted anyway.
    """

    __slots__ = ("run_id", "started_at", "finished_at", "token", "status",
                 "error", "_entries", "_chars", "_truncated", "_cond")

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.started_at = utcnow()
        self.finished_at: str | None = None
        # The run names the token, so a call abandoned deep in the pipeline
        # still says in its message which run it belonged to.
        self.token = CancelToken(f"report {run_id}",
                                 reason="the operator stopped it")
        self.status = RUNNING
        self.error: str | None = None
        self._entries: list[tuple[str, Any]] = []
        self._chars = 0
        self._truncated = False
        # Guards the entry list and the status, and wakes every watcher parked
        # on it.  One condition for both, because a watcher waiting for more
        # entries is equally waiting to find out there will not be any.
        self._cond = threading.Condition()

    # -- writing ----------------------------------------------------------------

    def append(self, kind: str, data: Any) -> None:
        with self._cond:
            if kind == "token" and isinstance(data, str):
                if self._truncated:
                    return
                self._chars += len(data)
                if self._chars > MAX_TEXT_CHARS:
                    self._truncated = True
                    self._entries.append((
                        "error",
                        f"The report passed {MAX_TEXT_CHARS // (1024 * 1024)}MB "
                        f"of text and the rest is not being kept. The run is "
                        f"still going and will still be saved."))
                    self._cond.notify_all()
                    log.error("run %s exceeded the text buffer at %d chars",
                              self.run_id, self._chars)
                    return
            self._entries.append((kind, data))
            self._cond.notify_all()

    def _settle(self, status: str, error: str | None = None) -> None:
        with self._cond:
            self.status = status
            self.error = error
            self.finished_at = utcnow()
            # The producer says "done" or "error" in its own words, which the
            # page reads for the report id and the failure text.  This says
            # only that there will be nothing more, which is the one thing a
            # watcher cannot work out for itself and the signal it stops on.
            self._entries.append(("ended", {"status": status, "error": error}))
            self._cond.notify_all()

    # -- reading ----------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        with self._cond:
            return len(self._entries)

    def snapshot(self) -> dict:
        with self._cond:
            return {"run_id": self.run_id, "status": self.status,
                    "started_at": self.started_at,
                    "finished_at": self.finished_at,
                    "error": self.error, "entries": len(self._entries)}

    def follow(self, cursor: int) -> Iterator[list[tuple[int, str, Any]] | None]:
        """Everything from cursor on, in batches, until the run has ended.

        A batch is whatever had accumulated when the watcher next looked, so a
        page attaching to a run that is already an hour old gets one enormous
        batch it can collapse into a few frames, while a page keeping up gets
        batches of one and sees each token as it is written.  Yielding None
        means nothing arrived and the connection should say something anyway.
        """
        cursor = max(0, cursor)
        while True:
            with self._cond:
                if cursor >= len(self._entries) and self.status == RUNNING:
                    self._cond.wait(KEEPALIVE_SECONDS)
                batch = [(i, kind, data) for i, (kind, data)
                         in enumerate(self._entries[cursor:], cursor)]
                ended = self.status != RUNNING
            if batch:
                cursor += len(batch)
                yield batch
                continue
            if ended:
                return
            yield None


# The run in flight, or the last one to finish.  Both live in the same slot
# because a page that reloads a moment after a report finished wants the same
# thing a page that reloads during one wants: the text, and then whether there
# is more coming.  Keeping the finished run until the next one starts is what
# makes those two the same request.
_current: Run | None = None
_lock = threading.Lock()


def start(producer: Callable[[], Iterable[tuple[str, Any]]]) -> Run:
    """Begin a run on its own thread.  Raises Busy if one is already going."""
    global _current
    with _lock:
        if _current is not None and _current.status == RUNNING:
            raise Busy(_current.run_id)
        run = Run(secrets.token_hex(8))
        _current = run

    def work() -> None:
        # The token is bound to this thread, not passed down: every pass builds
        # its own client deep inside its own loop, and current_cancel_token()
        # is how those calls find out whose run they are making.
        with cancellation_scope(run.token):
            try:
                for kind, data in producer():
                    run.append(kind, data)
            except Cancelled as exc:
                # Not an Exception, deliberately, so the bare "except Exception"
                # around each model call in the report passes stands aside for
                # it rather than filing an abort as one more failed draw.
                log.info("run %s stopped: %s", run.run_id, exc)
                run._settle("stopped", str(exc))
                return
            except Exception as exc:
                log.error("run %s failed: %s", run.run_id, exc)
                run.append("error", str(exc).splitlines()[0] or str(exc))
                run._settle("failed", str(exc))
                return
            run._settle("done")
            log.info("run %s finished", run.run_id)

    threading.Thread(target=work, name=f"report-{run.run_id}",
                     daemon=True).start()
    log.info("run %s started", run.run_id)
    return run


def active() -> Run | None:
    """The run being written right now, or None."""
    with _lock:
        if _current is not None and _current.status == RUNNING:
            return _current
        return None


def latest() -> Run | None:
    """The run in flight, or the last one to finish, or None."""
    with _lock:
        return _current


def get(run_id: str) -> Run | None:
    with _lock:
        if _current is not None and _current.run_id == run_id:
            return _current
        return None


def cancel(run_id: str) -> bool:
    """Abandon a run and the model call it is blocked inside.

    Aimed at the run named rather than at "whatever is running", so a stop
    issued against a run that has already ended finds a spent token and stops
    there instead of reaching into the run that replaced it.
    """
    run = get(run_id)
    if run is None or run.status != RUNNING:
        return False
    log.info("run %s: stopping it and the model call it is inside", run_id)
    return run.token.cancel()
