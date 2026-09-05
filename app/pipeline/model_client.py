"""Client for Docker Model Runner, shaped like the Ollama client it replaces.

Model Runner speaks the OpenAI dialect (/v1/chat/completions, /v1/embeddings,
/v1/models); every caller here was written against Ollama's /api/generate.
The translation lives in this one module: same class surface, same method
signatures, and generate() still returns a dict with a "response" key, so the
five calling modules did not have to change.

Two facts about Model Runner drive most of what follows.

First, the endpoint depends on where the process runs. From inside a container
it is plain HTTP at model-runner.docker.internal. From the host there is no TCP
listener at all - localhost is refused - and the Docker socket is the only way
in, so a unix-socket transport is included for host-side use.

Second: these are reasoning models and this API has no way to turn reasoning
off. Reasoning and answer share one token budget, and the reply separates them
as message.content and message.reasoning_content. When the budget runs out
mid-deliberation the reply has empty content, a full reasoning_content, and
finish_reason "length" - tokens were spent and nothing usable came back. That
exact failure cost this project hours under Ollama, where the deliberation was
invisible. Here it is detected and raised with the fix in the first line of the
message, because the callers surface only the first line.

Third, a call can outlive the reason it was made. The operator changes the
endpoint and presses re-ingest while the worker is minutes into a request built
from the settings they have just replaced, and the answer it is waiting for is
one nobody wants. Abandoning that call is therefore this module's job rather
than the caller's, because the socket it is blocked on is only reachable from
here: a run binds a CancelToken to its thread, requests issued on that thread
are made in a form another thread can interrupt, and an interrupted one raises
Cancelled - which is deliberately not a ModelRunnerError, because nothing
failed and there is nothing to retry.
"""
from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import random
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import requests

from .config import env_bool, env_float, env_int, env_str
from .log import get_logger

log = get_logger("model")


def same_model(a: str, b: str) -> bool:
    """Do these two names refer to the same model?

    Model Runner reports ids like docker.io/ai/gemma4:latest while an operator
    writes ai/gemma4, and a remote endpoint may report either. Registry prefix
    and a :latest tag are noise for this comparison. Module level because the
    settings page has to answer the same question about an endpoint it has not
    connected a client to yet.
    """
    def stem(name: str) -> str:
        name = (name or "").strip()
        for prefix in ("docker.io/", "registry-1.docker.io/"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.endswith(":latest"):
            name = name[:-len(":latest")]
        return name
    return stem(a) == stem(b)


class ModelRunnerError(RuntimeError):
    pass


class ModelNotSet(ModelRunnerError):
    pass


class ModelMissing(ModelRunnerError):
    pass


class ReasoningStarvation(ModelRunnerError):
    """All output tokens went to reasoning; the answer never arrived."""


# Callers catch these by their old names; aliases keep the swap surgical.
OllamaError = ModelRunnerError


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------
class Cancelled(BaseException):
    """The run this call belonged to was abandoned before the answer arrived.

    Deliberately not a ModelRunnerError, and deliberately not an Exception at
    all. Both stages that call a model in a loop wrap the call in a bare
    "except Exception" so that one unreadable page cannot sink a whole
    document, and an abandonment caught by one of those would be filed as a
    page error while the loop carried on through the remaining pages -
    finishing, slowly and expensively, a run that has already been replaced.
    Inheriting from BaseException is what makes those handlers stand aside, the
    same way they stand aside for KeyboardInterrupt, which is the same kind of
    event: not a failure of the request but an instruction to stop making it.

    The runner catches this by name at the top of its worker loop and around
    each stage, which is where an abandoned document is put back into a state
    the next run can redo.
    """


class CancelToken:
    """One run's claim on the model calls made for it.

    A cancel is addressed to a token rather than to "whatever is running now",
    because by the time it is delivered the run being cancelled has usually
    already been replaced: the operator presses re-ingest, the replacement is
    queued, and the abort arrives afterwards. A token belongs to one run, is
    never reused, and stays cancelled once cancelled - so a late abort finds a
    spent token and stops there instead of reaching into the run that took
    over.

    Every method here may be called from another thread while the thread that
    owns the token is blocked inside a request. That is the whole purpose of
    the class, and it is why the lock is held only long enough to take a copy
    of what has to be closed.
    """

    def __init__(self, label: str = ""):
        # label names the work in the exception message and nowhere else.
        self.label = label
        self._flag = threading.Event()
        self._lock = threading.Lock()
        # Events to set: they wake a thread parked on a call it is about to
        # walk away from.
        self._waiters: list[threading.Event] = []
        # Handles to close: sockets and responses. Setting a flag only ends
        # this side's waiting, while closing the socket also ends the request
        # that was being waited for.
        self._handles: list[Any] = []

    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()

    def cancel(self) -> bool:
        """Abandon this run's calls. True if this call is the one that did it.

        Safe when nothing is in flight and safe when the run has already
        finished: a token with nothing registered against it has nothing to
        close, and cancelling twice is a no-op the second time.
        """
        with self._lock:
            if self._flag.is_set():
                return False
            self._flag.set()
            waiters, handles = self._waiters[:], self._handles[:]
            self._waiters.clear()
            self._handles.clear()
        # Outside the lock deliberately: closing a socket can take a moment,
        # and a thread trying to register one must not queue behind that.
        for event in waiters:
            event.set()
        for handle in handles:
            _close_handle(handle)
        return True

    def sleep(self, seconds: float) -> None:
        """Wait, but no longer than it takes for this run to be abandoned."""
        self._flag.wait(seconds)

    def watch(self, handle: Any) -> bool:
        """Register a socket or response to close if this run is abandoned.

        False means the abort has already happened and this handle missed the
        sweep, so the caller must close it itself - no later sweep will.
        """
        with self._lock:
            if self._flag.is_set():
                return False
            self._handles.append(handle)
            return True

    def release(self, handle: Any) -> None:
        """Forget a handle that has been closed the ordinary way."""
        with self._lock:
            if handle in self._handles:
                self._handles.remove(handle)

    def park(self, event: threading.Event) -> bool:
        """Register an event to set on cancel, so a wait on it ends there."""
        with self._lock:
            if self._flag.is_set():
                return False
            self._waiters.append(event)
            return True

    def unpark(self, event: threading.Event) -> None:
        with self._lock:
            if event in self._waiters:
                self._waiters.remove(event)


_bound = threading.local()


def current_cancel_token() -> CancelToken | None:
    """The token this thread's model calls answer to, or None.

    Thread-local rather than an argument: every stage builds its own client
    deep inside its own loop, and threading a token down through five modules
    would touch every call site in the pipeline. It also gives the right
    default - a thread nobody bound a token to, such as one serving a chat
    request, is not cancellable at all, which is what keeps re-ingest from
    reaching into a conversation.
    """
    return getattr(_bound, "token", None)


@contextmanager
def cancellation_scope(token: CancelToken | None) -> Iterator[None]:
    """Bind a token to this thread for the duration of a run.

    The previous binding is restored rather than cleared, so a scope inside a
    scope leaves the outer run still cancellable.
    """
    previous = getattr(_bound, "token", None)
    _bound.token = token
    try:
        yield
    finally:
        _bound.token = previous


def raise_if_cancelled() -> None:
    """Stop here if the run on this thread has been abandoned."""
    token = current_cancel_token()
    if token is not None and token.cancelled:
        raise Cancelled(_cancel_message(token))


def _cancel_message(token: CancelToken | None) -> str:
    named = f" for {token.label}" if token is not None and token.label else ""
    return f"the model call{named} was abandoned: this run has been superseded"


def _socket_of(handle: Any) -> socket.socket | None:
    """Find the socket underneath a response, if it can still be reached.

    requests wraps urllib3 which wraps http.client, and which attribute holds
    the socket differs between their versions, so each known route is tried and
    a miss is not an error - closing the response is the fallback. It is only a
    slower one: a close does not reliably wake a thread already blocked in a
    read, and a shutdown does.
    """
    if isinstance(handle, socket.socket):
        return handle
    raw = getattr(handle, "raw", handle)
    for route in (("_connection", "sock"),
                  ("_fp", "fp", "raw", "_sock"),
                  ("_original_response", "fp", "raw", "_sock")):
        node: Any = raw
        for name in route:
            node = getattr(node, name, None)
            if node is None:
                break
        if isinstance(node, socket.socket):
            return node
    return None


def _close_handle(handle: Any) -> None:
    """Stop a request another thread is waiting on.

    shutdown() before close(), because closing a socket does not reliably wake
    a thread already blocked reading from it - on macOS it does not - while
    shutting it down makes that read return at once. Sockets are only shut
    down, never closed here: the code that opened one closes it, and closing a
    file descriptor out from under its owner hands its number to whatever opens
    the next one. Every step is best effort, because the only thing worse than
    a request that will not stop is an abort that raises half way through and
    leaves the rest of the handles open.
    """
    if handle is None:
        return
    sock = _socket_of(handle)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    if not isinstance(handle, socket.socket):
        try:
            handle.close()
        except Exception:
            pass


# How much of a response body is read at a time. Large enough that an ordinary
# completion arrives in one or two pieces, small enough that an abort lands
# inside a slow body rather than after it.
BODY_CHUNK = 65536

DEFAULT_URL = "http://model-runner.docker.internal/v1"
# The socket path prefix carries the Docker Desktop version and will move;
# overridable for that day rather than hard-coded until it breaks.
SOCKET_PREFIX = os.environ.get("MODEL_RUNNER_SOCKET_PREFIX", "/exp/vDD4.40/v1")

BIND_HINT = (
    "Cannot reach Docker Model Runner at {url}.\n"
    "\n"
    "Model Runner is a Docker Desktop feature and is off by default:\n"
    "\n"
    "    docker desktop enable model-runner\n"
    "    docker model ls\n"
    "\n"
    "From a container the endpoint is http://model-runner.docker.internal/v1 .\n"
    "From the host there is no TCP listener; set MODEL_RUNNER_SOCKET to the\n"
    "Docker socket (usually ~/.docker/run/docker.sock) instead."
)


def _normalize_url(url: str) -> str:
    """Guarantee exactly one /v1 with no trailing slash, wherever it came from.

    The env value and the constructor argument both pass through here, because
    a missing or doubled /v1 fails with a bare 404 that says nothing useful.
    """
    base = (url or "").strip().rstrip("/")
    if not base:
        return DEFAULT_URL
    if not base.endswith("/v1"):
        base += "/v1"
    return base


# --------------------------------------------------------------------------
# unix-socket transport (host-side only)
# --------------------------------------------------------------------------
class _UnixHTTP:
    """Minimal POST/GET over the Docker socket.

    Deliberately does NOT support streaming: assembling a trustworthy SSE
    reader over raw http.client means re-implementing chunked-transfer
    truncation detection, and the host-side users of this module (scripts,
    preflight checks) only ever need whole responses. stream() falls back to a
    single non-streamed request on this transport.
    """

    def __init__(self, socket_path: str):
        self.socket_path = os.path.expanduser(socket_path)

    def request(self, method: str, path: str, body: bytes | None,
                timeout: float) -> tuple[int, bytes]:
        token = current_cancel_token()
        conn = http.client.HTTPConnection("localhost", timeout=timeout)
        opened: list[socket.socket] = []

        def _connect():
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(self.socket_path)
            conn.sock = s
            opened.append(s)
            # Registered here rather than after the request returns, because by
            # then the wait this exists to interrupt is over. This is the one
            # transport whose socket this module opens itself, which is what
            # makes it abortable without a second thread to do the waiting.
            if token is not None and not token.watch(s):
                _close_handle(s)

        conn.connect = _connect  # type: ignore[method-assign]
        try:
            conn.request(method, path, body=body,
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read()
        except (OSError, http.client.HTTPException) as exc:
            # A socket shut down from another thread arrives here as a read
            # error. It is an abandonment rather than a fault of the endpoint,
            # and the retry loop above must not answer it by asking again.
            if token is not None and token.cancelled:
                raise Cancelled(_cancel_message(token)) from exc
            raise
        finally:
            if token is not None:
                for s in opened:
                    token.release(s)
            conn.close()
        # An abort that lands as the answer arrives still abandons the answer:
        # the run it was for is over, and returning it would carry work into a
        # document that is being written by somebody else now.
        if token is not None and token.cancelled:
            raise Cancelled(_cancel_message(token))
        return response.status, data


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, url: str | None = None, timeout: int | None = None,
                 retries: int | None = None, keep_alive: str | None = None,
                 api_key: str | None = None, allow_override: bool = True):
        # keep_alive is accepted and ignored: residency is Model Runner's own
        # business. Old env names are honoured so an existing .env keeps
        # working after the swap.
        del keep_alive
        # A caller that names its own url means it - embeddings do exactly
        # this, which is what keeps them local whatever the text model is
        # pointed at. Only a bare client consults the operator's choice, and
        # allow_override=False lets the settings page test an address without
        # the address it is testing being substituted underneath it.
        #
        # What that choice resolves to is the mode, not the presence of a
        # stored value: in local mode client_override() returns ("", "") and
        # everything below is the chain this class used before the settings
        # page existed, socket included. Its failures are deliberately not
        # caught. It raises when the operator has selected an endpoint that
        # cannot be used, and answering that with the local model would be the
        # substitution this whole mechanism exists to prevent.
        override_url, override_key = "", ""
        if url is None and allow_override:
            from .llm_settings import client_override
            override_url, override_key = client_override()
        # Whether the address came from the operator rather than from .env or
        # the built-in default. require_model needs it: the remedy for a model
        # this endpoint does not serve is a different remedy in each case.
        self.from_settings = bool(override_url)
        raw_url = (url or override_url
                   or env_str("MODEL_URL", "") or env_str("OLLAMA_URL", ""))
        self.url = _normalize_url(raw_url or DEFAULT_URL)
        self.api_key = api_key if api_key is not None else override_key
        self.timeout = timeout if timeout is not None else (
            env_int("MODEL_TIMEOUT", 0) or env_int("OLLAMA_TIMEOUT", 1800))
        self.retries = max(1, retries if retries is not None else (
            env_int("MODEL_RETRIES", 0) or env_int("OLLAMA_RETRIES", 3)))
        # The unix socket reaches the local runner and nothing else, so a
        # client pointed anywhere else must not use it - it would ignore the
        # address entirely and quietly answer from the local model.
        # Only an override sends this client somewhere the socket cannot
        # reach. A caller naming the local address explicitly - embeddings do,
        # transcription does - still wants the socket, and blanking it for them
        # quietly moved those calls onto TCP.
        # The socket reaches the local runner and nothing else, so it is kept
        # only when this client is actually pointed there. Deciding on "did an
        # override supply the url" was wrong in one direction that mattered:
        # the settings page tests a remote address by passing it explicitly,
        # and with the socket still set the test queried the local runner and
        # reported the remote endpoint healthy on the strength of it.
        local = _normalize_url(env_str("MODEL_URL", "")
                               or env_str("OLLAMA_URL", "") or DEFAULT_URL)
        self.socket_path = (os.environ.get("MODEL_RUNNER_SOCKET", "")
                            if self.url == local else "")
        self.session = requests.Session()

    # -- plumbing ------------------------------------------------------------

    def _headers(self) -> dict:
        """Request headers, carrying the key only when one is configured.

        The key is put here and nowhere else: not in the URL, not in a log
        line, not in an exception. A remote endpoint needs it; the local runner
        neither needs nor sees one.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: dict, *, stream: bool = False):
        """POST with retry on connection errors and 5xx.

        Returns the whole body, except for the streaming caller - the only one
        that needs the response object itself - which gets it open, and with it
        the duty of closing it and releasing it from the cancellation token.

        A read timeout is deliberately NOT retried: on a loaded local model a
        timeout is not transient, and retrying it triples the wasted wall
        clock before the user sees the real problem.

        Neither is an abandonment retried, and for a stronger reason. It is not
        a failure of anything; it is the operator saying this run is over, and
        a retry loop that answered it by asking again would keep the endpoint
        working on an answer with nowhere to go - which is the whole of what
        this mechanism exists to stop.
        """
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            token = current_cancel_token()
            raise_if_cancelled()
            try:
                if self.socket_path:
                    status, data = _UnixHTTP(self.socket_path).request(
                        "POST", f"{SOCKET_PREFIX}{path}", body, self.timeout)
                    if status >= 500:
                        raise ModelRunnerError(
                            f"Model Runner returned {status}: {data[:300].decode(errors='replace')}")
                    if status >= 400:
                        self._raise_api_error(status, data)
                    return data
                response = self._send(path, body, token)
                # Handing the response to the token is what lets an abort that
                # lands while the body is still arriving close the socket,
                # rather than leaving this thread reading an answer the run it
                # belongs to has already stopped wanting.
                if token is not None and not token.watch(response):
                    _close_handle(response)
                    raise Cancelled(_cancel_message(token))
                streaming = False
                try:
                    if response.status_code >= 500:
                        raise ModelRunnerError(
                            f"Model Runner returned {response.status_code}: "
                            f"{self._read(response, token)[:300].decode(errors='replace')}")
                    if response.status_code >= 400:
                        self._raise_api_error(response.status_code,
                                              self._read(response, token))
                    if stream:
                        streaming = True
                        return response
                    return self._read(response, token)
                finally:
                    if not streaming:
                        if token is not None:
                            token.release(response)
                        response.close()
            except requests.exceptions.ReadTimeout as exc:
                raise ModelRunnerError(
                    f"the model did not answer within {self.timeout}s - raise "
                    f"OLLAMA_TIMEOUT if pages are genuinely this slow") from exc
            except (requests.exceptions.ConnectionError, OSError,
                    http.client.HTTPException) as exc:
                # A connection torn down by an abort looks exactly like one the
                # network dropped, and retrying it would restart the very call
                # that was abandoned.
                if token is not None and token.cancelled:
                    raise Cancelled(_cancel_message(token)) from exc
                last = exc
                if attempt < self.retries:
                    self._backoff(attempt, token)
            except ModelRunnerError as exc:
                last = exc
                if attempt < self.retries and "returned 5" in str(exc):
                    self._backoff(attempt, token)
                else:
                    raise
        if isinstance(last, ModelRunnerError):
            raise last
        raise ModelRunnerError(BIND_HINT.format(url=self.url)) from last

    def _backoff(self, attempt: int, token: CancelToken | None) -> None:
        """Wait before trying again - but not through an abort.

        Fifteen seconds of sleep is fifteen seconds of a button that looks
        broken, so a run that has been abandoned wakes out of the backoff
        instead of serving it out and then discovering there was nothing to
        retry for.
        """
        delay = min(2 ** attempt, 15)
        if token is not None:
            token.sleep(delay)
        else:
            time.sleep(delay)

    def _send(self, path: str, body: bytes, token: CancelToken | None):
        """Issue the request and return the response as its headers arrive.

        stream=True goes on the wire whatever the caller asked for. It is not a
        change of protocol - the payload still says whether the model should
        stream - only of who reads the body: requests hands it back unread,
        which puts it under _read() where an abort can land part way through
        it. A caller with no token is left on the path it has always taken,
        because a call nobody can cancel gains nothing from being interruptible
        and this module has no business spending a thread on it.
        """
        if token is None:
            return self.session.post(
                f"{self.url}{path}", data=body, stream=True,
                headers=self._headers(), timeout=self.timeout)
        return self._send_cancellable(path, body, token)

    def _send_cancellable(self, path: str, body: bytes, token: CancelToken):
        """Wait for the response on a thread this one can walk away from.

        Nothing about a request is reachable from outside until its headers
        arrive: the socket does not exist when the call starts, so there is no
        handle for another thread to close, and a flag alone cannot end a
        blocking read. A non-streamed completion sends no headers until the
        model has finished, and on a remote 31B that is precisely the
        minutes-long wait an operator is trying to interrupt. Doing the waiting
        on a thread of its own is what lets this one stop waiting at once.

        What is left behind is one daemon thread, still inside the request. It
        closes whatever it is eventually handed and ends there. The endpoint
        goes on producing an answer nobody will read, which is the honest cost
        of not being able to reach its socket; it delays nothing here, because
        the run that replaced this one is already moving.
        """
        finished = threading.Event()
        outcome: dict[str, Any] = {}

        def call() -> None:
            try:
                outcome["response"] = self.session.post(
                    f"{self.url}{path}", data=body, stream=True,
                    headers=self._headers(), timeout=self.timeout)
            except BaseException as exc:   # re-raised on the waiting thread
                outcome["error"] = exc
            finally:
                finished.set()
                # The waiter may be long gone. Nobody else will close this, and
                # an unclosed response holds its connection open. Both sides
                # take the response with pop(), so exactly one of them gets it.
                if token.cancelled:
                    _close_handle(outcome.pop("response", None))

        threading.Thread(target=call, name="model-request", daemon=True).start()
        # park() fails only when the abort has already happened, in which case
        # there is nothing left to wait for.
        if token.park(finished):
            try:
                finished.wait()
            finally:
                token.unpark(finished)
        if token.cancelled:
            _close_handle(outcome.pop("response", None))
            raise Cancelled(_cancel_message(token))
        error = outcome.get("error")
        if error is not None:
            raise error
        response = outcome.pop("response", None)
        if response is None:
            # The abort landed between this thread waking and taking the
            # response, and the request thread cleared it away.
            raise Cancelled(_cancel_message(token))
        return response

    def _read(self, response, token: CancelToken | None) -> bytes:
        """Read a whole body, in pieces, so an abort can land inside one.

        Asking requests for the body in one call would be one more wait with no
        way out of it. In pieces this costs nothing on the ordinary path, where
        a completion arrives in one or two of them, and it closes the response
        either way - the connection goes back to the pool exactly once.
        """
        pieces: list[bytes] = []
        try:
            for piece in response.iter_content(BODY_CHUNK):
                if token is not None and token.cancelled:
                    raise Cancelled(_cancel_message(token))
                if piece:
                    pieces.append(piece)
        except (requests.exceptions.RequestException, OSError) as exc:
            # A socket closed under a read is what an abort looks like from in
            # here. Anything else is a real transport failure and keeps the
            # handling it has always had, one frame up.
            if token is not None and token.cancelled:
                raise Cancelled(_cancel_message(token)) from exc
            raise
        finally:
            response.close()
        return b"".join(pieces)

    @staticmethod
    def _raise_api_error(status: int, body: bytes) -> None:
        try:
            detail = json.loads(body)
            message = (detail.get("error") or {}).get("message") or str(detail)
        except Exception:
            message = body[:300].decode(errors="replace")
        raise ModelRunnerError(f"Model Runner rejected the request ({status}): {message}")

    @staticmethod
    def _parse_json(raw) -> dict:
        """Bodies are parsed defensively: an HTML error page from a proxy or a
        half-written body must surface as our error type, not a JSONDecodeError
        five frames up."""
        data = raw if isinstance(raw, (bytes, str)) else raw.content
        try:
            return json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            head = (data[:200] if isinstance(data, (bytes, bytearray)) else str(data)[:200])
            raise ModelRunnerError(f"Model Runner sent a non-JSON reply: {head!r}") from exc

    # -- introspection --------------------------------------------------------

    def list_models(self) -> list[str]:
        # A listing is short and bounded by its own 30s timeout, so it is left
        # as it was rather than made interruptible; this check only stops an
        # already abandoned run from starting one.
        raise_if_cancelled()
        try:
            if self.socket_path:
                status, data = _UnixHTTP(self.socket_path).request(
                    "GET", f"{SOCKET_PREFIX}/models", None, 30)
                if status != 200:
                    raise ModelRunnerError(f"/models returned {status}")
            else:
                response = self.session.get(f"{self.url}/models", timeout=30,
                                            headers=self._headers())
                response.raise_for_status()
                data = response.content
        except requests.exceptions.HTTPError as exc:
            # The server answered and refused, so it is reachable. Reporting
            # that as "cannot reach" sends an operator to look at networking
            # when the answer is in the reply: 401 is the key, 403 the
            # permission, 404 usually a base URL missing its /v1.
            status = (exc.response.status_code
                      if exc.response is not None else "an error")
            raise ModelRunnerError(
                f"{self.url} answered {status}. The address is reachable, so "
                f"check the model endpoint settings - a 401 means the API key, "
                f"a 404 usually means the address needs its /v1 suffix."
            ) from exc
        except (requests.exceptions.RequestException, OSError) as exc:
            raise ModelRunnerError(BIND_HINT.format(url=self.url)) from exc
        parsed = self._parse_json(data)
        return sorted(m.get("id", "") for m in parsed.get("data", []) if m.get("id"))

    @staticmethod
    def _same_model(a: str, b: str) -> bool:
        return same_model(a, b)

    def _is_local_runner(self) -> bool:
        """Is this the built-in Model Runner, or a server somebody else owns?

        The socket reaches the local runner and nothing else, and the default
        host name resolves to it from inside a container. Any other address is
        one an operator chose, where a pull on this machine adds nothing.
        """
        return bool(self.socket_path) or "model-runner.docker.internal" in self.url

    def _how_to_get(self, model: str) -> str:
        """The remedy for a model this endpoint does not serve.

        "docker model pull" is the answer for the built-in runner and is wrong
        everywhere else: an operator cannot pull a model into a server they do
        not administer, and following that advice would leave them pulling
        onto this machine while the request keeps going elsewhere.
        """
        if self.from_settings:
            return (f"Pick a model {self.url} already serves, on the settings "
                    f"page, or point that page at an endpoint serving {model}.")
        if self._is_local_runner():
            return f"Pull it first:  docker model pull {model}"
        return (f"{self.url} is not the built-in runner, so 'docker model "
                f"pull' will not add {model} to it - name a model it serves, "
                f"or point MODEL_URL at an endpoint that has {model}.")

    def require_model(self, model: str, var_name: str) -> str:
        """Check a model name against what this endpoint actually serves.

        var_name is a label for where the value came from, and is not
        necessarily an environment variable: the same text pass reads its
        model from the settings page on one machine and from .env on another,
        and an error naming the wrong one sends the operator to the wrong
        place to fix it. The first line stands alone, because the callers of
        this pipeline print only the first line.
        """
        model = (model or "").strip()
        installed = self.list_models()
        offers = (", ".join(installed[:6]) + (" ..." if len(installed) > 6 else "")
                  if installed else "")
        listing = ("\n  ".join(installed) if installed
                   else "(none - this endpoint lists no models at all)")
        if not model:
            raise ModelNotSet(
                f"{var_name} is not set, so nothing can be asked of "
                f"{self.url}.\n"
                + ("Choose one on the settings page.\n" if self.from_settings
                   else "Set it in .env and restart.\n")
                + f"{self.url} serves:\n  " + listing)
        if not any(self._same_model(model, have) for have in installed):
            # The first line already names up to six; spelling the list out
            # again below only earns its space when it was truncated there.
            full = (f"\n{self.url} serves:\n  " + listing
                    if len(installed) > 6 else "")
            raise ModelMissing(
                f"{var_name} is {model}, but {self.url} does not serve it"
                + (f" - it offers {offers}." if offers
                   else " and lists no models at all.")
                + full + "\n\n" + self._how_to_get(model))
        return model

    # -- message building ------------------------------------------------------

    @staticmethod
    def _image_part(path: Path) -> dict:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    def _messages(self, prompt: str, system: str | None,
                  images: list[Path] | None) -> list[dict]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            content += [self._image_part(p) for p in images]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _payload(model: str, messages: list[dict], options: dict | None,
                 format_json: bool, stream: bool,
                 think: bool | None = None) -> dict:
        options = options or {}
        payload: dict[str, Any] = {"model": model, "messages": messages,
                                   "stream": stream}
        # Reasoning is on by default and shares the output budget; with the
        # caps at 0 that means unbounded deliberation - measured at five
        # minutes for a two-sentence page. llama.cpp honours this template
        # switch (measured: 31 tokens with reasoning, 3 without, same answer),
        # where the OpenAI-style reasoning_effort knob is silently ignored.
        enabled = thinking_enabled() if think is None else think
        if not enabled:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if "temperature" in options:
            payload["temperature"] = options["temperature"]
        if "seed" in options:
            payload["seed"] = options["seed"]
        # num_predict <= 0 means uncapped: max_tokens is simply omitted.
        limit = options.get("num_predict")
        if isinstance(limit, int) and limit > 0:
            payload["max_tokens"] = limit
        # num_ctx has no equivalent here - the server sizes its own context.
        if format_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _starvation(model: str, reasoning: str, finish: str | None,
                    usage: dict) -> ReasoningStarvation:
        spent = usage.get("completion_tokens")
        spent_text = f"{spent} output tokens" if spent else "its output budget"
        # First line self-contained: callers print only the first line.
        return ReasoningStarvation(
            f"{model} spent {spent_text} reasoning and returned no answer - "
            f"raise the *_NUM_PREDICT cap (0 = uncapped) or the timeout. "
            f"(finish_reason={finish}, reasoning began: "
            f"{reasoning.strip()[:120]!r})")

    # -- generation --------------------------------------------------------------

    def generate(self, model: str, prompt: str, *, images: list[Path] | None = None,
                 system: str | None = None, options: dict[str, Any] | None = None,
                 format_json: bool = False, think: bool | None = None) -> dict:
        payload = self._payload(model, self._messages(prompt, system, images),
                                options, format_json, stream=False, think=think)
        raw = self._post("/chat/completions", payload)
        parsed = self._parse_json(raw)

        choices = parsed.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        content = (message.get("content") or "").strip()
        reasoning = message.get("reasoning_content") or ""
        finish = choices[0].get("finish_reason") if choices else None
        usage = parsed.get("usage") or {}

        if reasoning:
            log.info("%s reasoned before answering (%s completion tokens total)",
                     model, usage.get("completion_tokens", "?"))

        if not content:
            if reasoning:
                # One retry helps only when a cap caused this; uncapped, the
                # same request would fail the same way.
                limit = (options or {}).get("num_predict") or 0
                if isinstance(limit, int) and limit > 0:
                    bigger = dict(options or {})
                    bigger["num_predict"] = min(limit * 4, 16384)
                    log.warning("%s starved its answer at %d tokens; retrying "
                                "once at %d", model, limit, bigger["num_predict"])
                    retry = self._payload(
                        model, self._messages(prompt, system, images),
                        bigger, format_json, stream=False, think=think)
                    parsed = self._parse_json(self._post("/chat/completions", retry))
                    choices = parsed.get("choices") or []
                    message = (choices[0].get("message") or {}) if choices else {}
                    content = (message.get("content") or "").strip()
                    reasoning = message.get("reasoning_content") or reasoning
                    finish = choices[0].get("finish_reason") if choices else finish
                    usage = parsed.get("usage") or usage
                if not content:
                    raise self._starvation(model, reasoning, finish, usage)
            else:
                raise ModelRunnerError(
                    f"{model} returned an empty reply (finish_reason={finish})")

        return {
            "response": content,
            "reasoning_content": reasoning or None,
            "eval_count": usage.get("completion_tokens"),
            "prompt_eval_count": usage.get("prompt_tokens"),
            "done_reason": finish,
        }

    # -- embeddings ---------------------------------------------------------------

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        BATCH = 64
        for start in range(0, len(inputs), BATCH):
            chunk = inputs[start:start + BATCH]
            parsed = self._parse_json(self._post(
                "/embeddings", {"model": model, "input": chunk}))
            rows = parsed.get("data") or []
            if len(rows) != len(chunk):
                raise ModelRunnerError(
                    f"asked for {len(chunk)} embeddings, got {len(rows)}")
            # Order is guaranteed by index, not by list position.
            rows.sort(key=lambda r: r.get("index", 0))
            vectors.extend(r["embedding"] for r in rows)
        return vectors

    # -- streaming ------------------------------------------------------------------

    def stream(self, model: str, prompt: str, *, system: str | None = None,
               options: dict[str, Any] | None = None,
               think: bool | None = None) -> Iterator[str]:
        if self.socket_path:
            # See _UnixHTTP: no SSE over the socket. One whole answer instead.
            yield self.generate(model, prompt, system=system,
                                options=options)["response"]
            return

        payload = self._payload(model, self._messages(prompt, system, None),
                                options, False, stream=True, think=think)
        # Captured before the request rather than read at each step, because a
        # generator runs on whichever thread pulls from it and this one is to
        # answer to the run that started it.
        token = current_cancel_token()
        response = self._post("/chat/completions", payload, stream=True)
        # Model Runner sends no charset, and requests then decodes text/* as
        # latin-1 per the HTTP RFC - every em dash becomes "\xe2\x80\x94" read
        # one byte at a time. The body is UTF-8; say so before decoding.
        response.encoding = "utf-8"

        emitted = ""              # actual answer text sent to the caller
        reasoned = 0              # reasoning fragments observed
        finish: str | None = None
        finished = False          # saw [DONE] - anything less is a cut stream
        error_frame: str | None = None

        try:
            for line in response.iter_lines(decode_unicode=True):
                # Between frames is where an abort lands on this path: the
                # answer arrives in many small pieces, so the wait between two
                # of them is short even when the whole answer is long.
                if token is not None and token.cancelled:
                    raise Cancelled(_cancel_message(token))
                if line is None or line == "" or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished = True
                    break
                try:
                    frame = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # An error can arrive as a 200-status SSE frame; swallowing it
                # would end the stream with silence and no diagnosis.
                if frame.get("error"):
                    error_frame = str((frame["error"] or {}).get("message")
                                      or frame["error"])[:300]
                    break
                choices = frame.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
                if delta.get("reasoning_content"):
                    reasoned += 1
                piece = delta.get("content")
                if piece:
                    emitted += piece
                    yield piece
        except (requests.exceptions.RequestException, OSError) as exc:
            # A stream whose socket was closed from another thread breaks
            # exactly like one the network cut. Reporting an abandonment as a
            # broken stream would put a fault on the endpoint for doing as it
            # was told, and would be caught by callers that go on to the next
            # page as though this one had merely failed.
            if token is not None and token.cancelled:
                raise Cancelled(_cancel_message(token)) from exc
            if emitted.strip():
                # The caller already has real text; die loudly, not silently.
                raise ModelRunnerError(
                    f"the stream broke mid-answer: {exc}") from exc
            raise ModelRunnerError(BIND_HINT.format(url=self.url)) from exc
        finally:
            # _post hands the streaming caller both the response and the duty
            # of putting it down, token included: a closed response left
            # registered would be closed again by an abort arriving later.
            if token is not None:
                token.release(response)
            response.close()

        if error_frame:
            raise ModelRunnerError(f"Model Runner reported an error mid-stream: {error_frame}")
        if emitted.strip():
            if not finished:
                raise ModelRunnerError(
                    "the stream ended without its [DONE] marker - the answer "
                    "may be incomplete")
            return
        # Nothing usable was emitted. Name the reason; never end in silence.
        if reasoned:
            raise self._starvation(model, "(streamed)", finish, {})
        if not finished:
            raise ModelRunnerError(
                "the stream ended before any answer arrived (connection cut "
                "or server stopped) - see 'docker model status'")
        raise ModelRunnerError(
            f"{model} finished ({finish}) without producing any text")


# Old name, same object: the five calling modules keep their import lines.
Ollama = ModelRunner


def random_seed() -> int:
    """A fresh seed, for callers that want independent samples.

    Sampling alone does not vary the answer: the same seed at the same
    temperature reproduces the same text. Varying the seed is what makes a
    second run an independent draw rather than a copy of the first.
    """
    return random.randrange(1, 2**31 - 1)


def default_options(temperature_var: str, ctx_var: str,
                    predict_var: str | None = None,
                    predict_default: int = 0,
                    seed: int | None = None) -> dict[str, Any]:
    """Reproducible by default; pass seed to draw an independent sample."""
    options = {
        "temperature": env_float(temperature_var, 0.0),
        "num_ctx": env_int(ctx_var, 8192),   # kept for callers; unused here
        "seed": env_int("OLLAMA_SEED", 42) if seed is None else int(seed),
    }
    # 0 or less means no cap; the request timeout is the backstop. On a
    # reasoning model a cap lands on the answer, not the deliberation.
    if predict_var:
        limit = env_int(predict_var, predict_default)
        if limit > 0:
            options["num_predict"] = limit
    return options


def thinking_enabled() -> bool:
    """Off by default: the pipeline wants the answer, not the deliberation."""
    return env_bool("MODEL_THINKING", False)
