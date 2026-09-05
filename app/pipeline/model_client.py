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
"""
from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import socket
import time
from pathlib import Path
from typing import Any, Iterator

import requests

from .config import env_bool, env_float, env_int, env_str
from .log import get_logger

log = get_logger("model")


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
        conn = http.client.HTTPConnection("localhost", timeout=timeout)

        def _connect():
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(self.socket_path)
            conn.sock = s

        conn.connect = _connect  # type: ignore[method-assign]
        try:
            conn.request(method, path, body=body,
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, url: str | None = None, timeout: int | None = None,
                 retries: int | None = None, keep_alive: str | None = None):
        # keep_alive is accepted and ignored: residency is Model Runner's own
        # business. Old env names are honoured so an existing .env keeps
        # working after the swap.
        del keep_alive
        raw_url = url or env_str("MODEL_URL", "") or env_str("OLLAMA_URL", "")
        self.url = _normalize_url(raw_url or DEFAULT_URL)
        self.timeout = timeout if timeout is not None else (
            env_int("MODEL_TIMEOUT", 0) or env_int("OLLAMA_TIMEOUT", 1800))
        self.retries = max(1, retries if retries is not None else (
            env_int("MODEL_RETRIES", 0) or env_int("OLLAMA_RETRIES", 3)))
        self.socket_path = os.environ.get("MODEL_RUNNER_SOCKET", "")
        self.session = requests.Session()

    # -- plumbing ------------------------------------------------------------

    def _post(self, path: str, payload: dict, *, stream: bool = False):
        """POST with retry on connection errors and 5xx.

        A read timeout is deliberately NOT retried: on a loaded local model a
        timeout is not transient, and retrying it triples the wasted wall
        clock before the user sees the real problem.
        """
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
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
                response = self.session.post(
                    f"{self.url}{path}", data=body, stream=stream,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout)
                if response.status_code >= 500:
                    response.close()
                    raise ModelRunnerError(
                        f"Model Runner returned {response.status_code}: "
                        f"{response.text[:300]}")
                if response.status_code >= 400:
                    data = response.content
                    response.close()
                    self._raise_api_error(response.status_code, data)
                return response
            except requests.exceptions.ReadTimeout as exc:
                raise ModelRunnerError(
                    f"the model did not answer within {self.timeout}s - raise "
                    f"OLLAMA_TIMEOUT if pages are genuinely this slow") from exc
            except (requests.exceptions.ConnectionError, OSError,
                    http.client.HTTPException) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 15))
            except ModelRunnerError as exc:
                last = exc
                if attempt < self.retries and "returned 5" in str(exc):
                    time.sleep(min(2 ** attempt, 15))
                else:
                    raise
        if isinstance(last, ModelRunnerError):
            raise last
        raise ModelRunnerError(BIND_HINT.format(url=self.url)) from last

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
        try:
            if self.socket_path:
                status, data = _UnixHTTP(self.socket_path).request(
                    "GET", f"{SOCKET_PREFIX}/models", None, 30)
                if status != 200:
                    raise ModelRunnerError(f"/models returned {status}")
            else:
                response = self.session.get(f"{self.url}/models", timeout=30)
                response.raise_for_status()
                data = response.content
        except (requests.exceptions.RequestException, OSError) as exc:
            raise ModelRunnerError(BIND_HINT.format(url=self.url)) from exc
        parsed = self._parse_json(data)
        return sorted(m.get("id", "") for m in parsed.get("data", []) if m.get("id"))

    @staticmethod
    def _same_model(a: str, b: str) -> bool:
        """Model Runner reports ids like docker.io/ai/gemma4:latest while the
        user writes ai/gemma4. Registry prefix and :latest are noise here."""
        def stem(name: str) -> str:
            name = name.strip()
            for prefix in ("docker.io/", "registry-1.docker.io/"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            if name.endswith(":latest"):
                name = name[:-7]
            return name
        return stem(a) == stem(b)

    def require_model(self, model: str, var_name: str) -> str:
        model = (model or "").strip()
        installed = self.list_models()
        if not model:
            raise ModelNotSet(
                f"{var_name} is not set in .env.\n"
                f"Models currently available:\n  "
                + ("\n  ".join(installed) if installed
                   else "(none - run 'docker model pull ai/gemma4')"))
        if not any(self._same_model(model, have) for have in installed):
            raise ModelMissing(
                f"{var_name}={model} is not pulled.\n"
                f"Available:\n  "
                + ("\n  ".join(installed) if installed else "(none)")
                + f"\n\nPull it first:  docker model pull {model}")
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
            if emitted.strip():
                # The caller already has real text; die loudly, not silently.
                raise ModelRunnerError(
                    f"the stream broke mid-answer: {exc}") from exc
            raise ModelRunnerError(BIND_HINT.format(url=self.url)) from exc
        finally:
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


def default_options(temperature_var: str, ctx_var: str,
                    predict_var: str | None = None,
                    predict_default: int = 0) -> dict[str, Any]:
    """Deterministic by default - transcription must be reproducible."""
    options = {
        "temperature": env_float(temperature_var, 0.0),
        "num_ctx": env_int(ctx_var, 8192),   # kept for callers; unused here
        "seed": env_int("OLLAMA_SEED", 42),
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
