"""Where the text model runs: endpoint, model name and key, in one place.

By default this pipeline talks to a model on this machine through Docker Model
Runner. An operator with a bigger machine on the network, or an account with a
hosted OpenAI-compatible API, can point the TEXT model somewhere else from the
settings page. This module is the only thing that knows how that choice is
stored and how it is resolved, so every other module asks rather than guesses.

An explicit mode, not the presence of a value, decides which model runs. The
operator picks "local" - the prepackaged model on the built-in runner - or
"external" - their own endpoint. Local mode ignores the stored endpoint, key
and model entirely, so switching back is one click and does not mean clearing
three fields; external mode uses them and fails loudly if they are not usable.

Three rules shape what follows.

The first is that the override belongs to the text model alone. Embeddings stay
on this machine, and nothing here is reachable from the embedding path: the
resolver is consulted only by a ModelRunner that was built with no endpoint of
its own, and embed.py always passes one and passes allow_override=False besides.

The second is that the API key is a secret held in the local SQLite settings
table. It leaves this module by one route only - client_override(), which the
model client uses to build an Authorization header. The configuration object
handed to the web layer carries a mask and a boolean instead of the key, so a
template cannot render the secret even by accident.

The third is that an unset override must mean exactly the behaviour that came
before it: the environment first, then the built-in local default. A
misconfigured override, by contrast, must fail loudly. An operator who believes
case text is going to a remote model, and is quietly still running locally, is
worse off than an operator looking at an error.

The vision model - the one that reads a scan or a photograph - is settled here
too, and separately. It is a different model and may live somewhere else
entirely: an operator who moves the text model to their own server usually has
not moved anything else, and the first thing they saw when they tried was a
transcription failure naming VLM_MODEL, an environment variable they had never
set. So vision has its own model name and its own choice of where it runs, and
that choice defaults to the local runner, which is what transcription has
always used. Where the endpoint can be asked - Ollama's native API answers
/api/show with a capability list - the choice is verified rather than believed,
because a text-only model handed a page image does not fail: it writes a
plausible page that then becomes evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from . import state
from .config import env_str
from .model_client import (DEFAULT_URL, FLAVOR_OLLAMA, FLAVOR_OPENAI,
                           ModelRunner, ModelRunnerError, same_model)

# The setting keys, named once. The web layer imports these rather than
# spelling the strings again, because a typo in a key name reads as "no
# override configured" and would fail silently.
SETTING_BASE_URL = "llm_base_url"
SETTING_MODEL = "llm_model"
SETTING_API_KEY = "llm_api_key"
SETTING_MODE = "llm_mode"
SETTING_API_FLAVOR = "llm_api_flavor"
# Vision has its own two keys for the same reason it has its own section
# below: it is a separate model, and the operator may want it in a separate
# place. The scope names where it runs; the model names which one to ask.
SETTING_VISION_MODEL = "llm_vision_model"
SETTING_VISION_SCOPE = "llm_vision_scope"

# The two modes, named once for the same reason as the keys. Local is the
# behaviour that existed before the settings page, and it is what an unset or
# unrecognised value means, so a database written by an older build - or by a
# future one that learns a third mode - degrades to the safe reading.
MODE_LOCAL = "local"
MODE_EXTERNAL = "external"
MODES = (MODE_LOCAL, MODE_EXTERNAL)

# Where the vision model runs. The same two words as the text mode, and
# deliberately the same two values, because an operator reading the page
# should not have to learn that "external" means one thing in one row and
# something else in the next. They are separate names rather than a reuse of
# MODE_* so that a future third text mode cannot silently become a third place
# to run vision. Local is the default and is today's behaviour exactly.
VISION_LOCAL = MODE_LOCAL
VISION_EXTERNAL = MODE_EXTERNAL
VISION_SCOPES = (VISION_LOCAL, VISION_EXTERNAL)

# Which dialect the external endpoint speaks. The names come from the model
# client, which is where the two request shapes are actually built, so there is
# one spelling of each word in the codebase rather than two that can drift.
#
# This exists because Ollama serves both. Its OpenAI-compatible /v1 endpoint
# accepts a request and then silently discards the options it has no field for
# - num_ctx among them - so a 6000-character chunk met a 4k default context,
# the model filled the window deliberating, and every call came back with
# finish_reason "length" and no content. Its native /api/chat takes those same
# options in an "options" object and honours them. The operator knows which
# kind of server they typed the address of; nothing here can reliably tell from
# the address alone, so it is asked rather than guessed.
FLAVORS = (FLAVOR_OPENAI, FLAVOR_OLLAMA)

# Limits exist to keep a paste accident out of the database and out of an HTTP
# header; they are not security boundaries.
MAX_URL = 500
MAX_MODEL = 200
MAX_KEY = 500

# The connectivity check runs while an operator watches a button, so it uses a
# short timeout and does not retry - a slow failure here is a failure.
CHECK_TIMEOUT = 20


class SettingsError(ValueError):
    """A value the operator typed cannot be stored as given."""


@dataclass(frozen=True)
class Resolved:
    """One effective value and where it came from.

    The source is what lets the settings page say "from settings" against a
    field the operator can clear, and "from environment" against one they
    cannot - which is the difference between a page that explains itself and a
    page that has to be read alongside the .env file.
    """

    value: str
    source: str          # "settings", "environment", "default" or "unset"

    @property
    def from_settings(self) -> bool:
        return self.source == "settings"


@dataclass(frozen=True)
class TextModelConfig:
    """The effective text-model configuration, safe to hand to a template.

    There is deliberately no key on this object. api_key_hint is a mask and
    api_key_set is a boolean, so the only thing a template can render is the
    fact that a key exists and roughly which one it is.
    """

    base_url: Resolved
    model: Resolved
    api_key_hint: str
    api_key_set: bool

    @property
    def overridden(self) -> bool:
        """True when the endpoint itself comes from the settings page.

        This is the flag the page should hang its warning on: a model name or a
        key alone changes nothing about where case text goes, but an endpoint
        does.
        """
        return self.base_url.from_settings


@dataclass
class CheckResult:
    """The outcome of one connectivity test, in terms an operator can act on.

    The key is not a field here and never appears in message: the test is run
    from a browser and its result goes back to that browser.
    """

    ok: bool
    message: str
    url: str = ""
    models: list[str] = field(default_factory=list)
    model_found: bool | None = None     # None when no model name was checked

    def as_dict(self) -> dict:
        return {"ok": self.ok, "message": self.message, "url": self.url,
                "models": self.models, "model_found": self.model_found}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _clean(value: str | None) -> str:
    return (value or "").strip()


def _setting(key: str) -> str:
    """Read one override, turning a database failure into our own error.

    A raw sqlite3 error escaping into the model client would surface five
    frames away as something that looks like a model problem. Naming the
    settings table here is what makes the real fault findable.
    """
    try:
        return _clean(state.get_setting(key, ""))
    except Exception as exc:                     # sqlite3.Error and friends
        raise ModelRunnerError(
            f"the model settings could not be read from the local database "
            f"({exc}) - the settings table is in the state database") from exc


def mode() -> str:
    """Which text model runs: the built-in local one, or the operator's own.

    Anything that is not exactly the external marker reads as local. That
    includes an unset key, so a database that predates this setting behaves as
    it always did, and it includes a value nothing here recognises, because
    guessing that an unknown word means "send the case text somewhere else"
    is the one wrong guess available.
    """
    return MODE_EXTERNAL if _setting(SETTING_MODE) == MODE_EXTERNAL else MODE_LOCAL


def is_external() -> bool:
    """True when the text passes should use the operator's endpoint."""
    return mode() == MODE_EXTERNAL


def set_mode(value: str) -> None:
    """Store the mode. This is the only writer, and it validates its input.

    An unrecognised value is refused rather than stored, because mode() would
    read it back as local and the operator would be told they were external
    while nothing external happened.
    """
    chosen = _clean(value).lower()
    if chosen not in MODES:
        raise SettingsError(
            f"the model mode must be {MODE_LOCAL!r} or {MODE_EXTERNAL!r} "
            f"(got {value!r})")
    state.set_setting(SETTING_MODE, chosen)


def stored_api_flavor() -> str:
    """The saved dialect whether or not the mode has it in force.

    The settings page needs this for the same reason it needs stored_base_url:
    the tickbox must show what would be used if the operator switched to
    external, or switching to local and back would look like the choice was
    thrown away.

    An unset or unrecognised value reads as the OpenAI dialect. That is the
    behaviour that existed before this setting, so a database written by an
    older build behaves exactly as it did, and a word nothing here recognises
    cannot quietly change the shape of every request.
    """
    return FLAVOR_OLLAMA if _setting(SETTING_API_FLAVOR) == FLAVOR_OLLAMA else FLAVOR_OPENAI


def api_flavor() -> str:
    """The dialect in force for the text model: "openai" or "ollama".

    The flavour belongs to the operator's own endpoint. Local mode is the
    built-in Model Runner, which speaks the OpenAI dialect and nothing else, so
    a tick left in the box from an earlier external configuration must not
    follow the operator back to the local model - it would send /api/chat to a
    server that has no such route and turn a working local pipeline into a run
    of 404s.
    """
    return stored_api_flavor() if is_external() else FLAVOR_OPENAI


def is_ollama_api() -> bool:
    """True when text calls should use Ollama's native API."""
    return api_flavor() == FLAVOR_OLLAMA


def set_api_flavor(value: str) -> None:
    """Store the dialect. This is the only writer, and it validates its input.

    Like set_mode, and for the same reason: a value the reader would not
    recognise is refused here rather than stored, because stored_api_flavor()
    reads anything unrecognised as the OpenAI dialect and the operator would be
    looking at an unticked box they had just ticked.

    It also has set_mode's shape rather than save_config's, because a tickbox
    has no empty-means-unchanged rule: it is posted on every save and its two
    states are both meaningful.
    """
    chosen = _clean(value).lower()
    if chosen not in FLAVORS:
        raise SettingsError(
            f"the API flavour must be {FLAVOR_OPENAI!r} or {FLAVOR_OLLAMA!r} "
            f"(got {value!r})")
    state.set_setting(SETTING_API_FLAVOR, chosen)


def _local_base_url() -> Resolved:
    """The endpoint with no override in play at all.

    This is exactly the chain ModelRunner used before the settings page
    existed, and it must stay that way: local mode has to be indistinguishable
    from the pipeline as it shipped.
    """
    for name in ("MODEL_URL", "OLLAMA_URL"):
        from_env = _clean(env_str(name, ""))
        if from_env:
            return Resolved(from_env, "environment")
    return Resolved(DEFAULT_URL, "default")


def base_url() -> Resolved:
    """The endpoint the text model uses, and where that address came from.

    Local mode does not consult the stored address even when one is stored:
    the mode is the decision, and a value left in the field is a value the
    operator may want back next week, not a value in force today.
    """
    if not is_external():
        return _local_base_url()
    stored = _setting(SETTING_BASE_URL)
    if stored:
        return Resolved(stored, "settings")
    # External mode with nothing stored is a misconfiguration, not a fallback.
    # Reporting the local default here would put the local address on the page
    # under a heading that says the operator's endpoint is in use.
    return Resolved("", "unset")


def text_model() -> Resolved:
    """The text model's name, which has to be overridable alongside the URL.

    Another endpoint serves differently-named models: the machine down the hall
    may call it qwen3-32b where this one calls it ai/qwen3. Without this the
    endpoint setting would be unusable on its own.
    """
    if is_external():
        stored = _setting(SETTING_MODEL)
        if stored:
            return Resolved(stored, "settings")
        # Deliberately no fall through to TEXT_MODEL here. The local model's
        # name on someone else's endpoint is the failure this mode exists to
        # prevent: it produced "TEXT_MODEL=ai/gemma4 is not pulled" from a
        # server that had never heard of ai/gemma4.
        return Resolved("", "unset")
    from_env = _clean(env_str("TEXT_MODEL", ""))
    if from_env:
        return Resolved(from_env, "environment")
    return Resolved("", "unset")


def stored_base_url() -> str:
    """The saved endpoint whether or not the mode has it in force.

    The settings page needs this to render its form: the fields must show what
    would be used if the operator switched to external, otherwise switching
    back looks like the values were thrown away.
    """
    return _setting(SETTING_BASE_URL)


def stored_text_model() -> str:
    """The saved model name whether or not the mode has it in force."""
    return _setting(SETTING_MODEL)


def _api_key() -> str:
    """The stored key. Module-private on purpose: see client_override()."""
    return _setting(SETTING_API_KEY)


def text_model_config() -> TextModelConfig:
    """Everything the settings page needs, and nothing it must not have."""
    key = _api_key()
    return TextModelConfig(base_url=base_url(), model=text_model(),
                           api_key_hint=mask_key(key), api_key_set=bool(key))


def client_override() -> tuple[str, str]:
    """The endpoint and key for a ModelRunner built with no url of its own.

    Local mode returns ("", ""), which is how the client falls back to the
    environment and then to the built-in runner, unix socket included. The
    stored endpoint and key are not read at all in that mode.

    External mode with no stored endpoint raises instead of returning empty.
    An empty return would send the request to the local runner, and an
    operator who believes a 31B model is answering while a 12B one actually is
    has nothing in the output to tell them apart.

    This is the only function that hands the key out, and its one caller is
    ModelRunner.__init__.
    """
    if not is_external():
        return "", ""
    url = _setting(SETTING_BASE_URL)
    if not url:
        raise ModelRunnerError(
            "the text model is set to an external endpoint, but no endpoint "
            "address is saved - enter one on the settings page, or switch the "
            "text model back to the local one")
    return url, _api_key()


def effective_text_model() -> str:
    """The model name callers should pass to generate/stream/require_model."""
    return text_model().value


def text_model_label() -> str:
    """How to name the source of the model name in an error message.

    require_model takes a label, and until now every caller passed the string
    "TEXT_MODEL". That sent an operator who had chosen a model on the settings
    page looking for an environment variable they never set. The label has to
    follow the value.
    """
    source = text_model().source
    if source == "settings":
        return "the text model chosen on the settings page"
    if source == "environment":
        return "TEXT_MODEL in .env"
    return "the text model"


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------
def mask_key(key: str) -> str:
    """A hint that identifies a key to the person who typed it, and no more.

    Enough characters to tell two keys apart at a glance; never enough to
    reconstruct one. A short key gets no characters at all, because on a short
    key even a four-character tail is a meaningful fraction of the secret.
    """
    key = _clean(key)
    if not key:
        return ""
    if len(key) < 12:
        return "(set)"
    return f"{key[:3]}...{key[-4:]}"


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def _validate_url(url: str) -> str:
    url = _clean(url)
    if not url:
        return ""
    if len(url) > MAX_URL:
        raise SettingsError(f"the endpoint address is longer than {MAX_URL} characters")
    if any(ch.isspace() for ch in url):
        raise SettingsError("the endpoint address cannot contain spaces")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise SettingsError(
            "the endpoint address must start with http:// or https:// "
            f"(got {parts.scheme or 'no scheme'})")
    if not parts.hostname:
        raise SettingsError("the endpoint address has no host")
    # A key belongs in the key field, not in the address. Storing
    # https://user:secret@host/v1 would put the secret in the settings table,
    # on the page and in every log line that names the endpoint, none of which
    # the key field's masking and redaction would cover.
    if parts.username or parts.password:
        raise SettingsError(
            "remove the username and password from the address and put the "
            "key in the API key field, where it is stored and shown as a "
            "secret")
    return url.rstrip("/")


def _validate_model(model: str) -> str:
    model = _clean(model)
    if not model:
        return ""
    if len(model) > MAX_MODEL:
        raise SettingsError(f"the model name is longer than {MAX_MODEL} characters")
    if any(ch.isspace() for ch in model):
        raise SettingsError("a model name cannot contain spaces")
    return model


def _validate_key(key: str) -> str:
    key = _clean(key)
    if not key:
        return ""
    if len(key) > MAX_KEY:
        raise SettingsError(f"the API key is longer than {MAX_KEY} characters")
    # The key becomes an HTTP header value. A stray newline from a paste would
    # be header injection, and a non-ASCII character raises inside the HTTP
    # library at request time - both are better refused here, where the error
    # can name the field. The message deliberately does not quote the value.
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) > 126 for ch in key):
        raise SettingsError(
            "the API key contains a space, a line break or a non-ASCII "
            "character - check for a stray newline in the paste")
    return key


def save(base_url_value: str, model_value: str, *,
         api_key_value: str | None = None, clear_api_key: bool = False) -> None:
    """Validate and store the override. An empty value clears that setting.

    The key follows the usual rule for a secret in a form: a blank field means
    "leave what is stored alone", because the field is rendered blank on every
    load. Removing a key is therefore a deliberate act - clear_api_key - and
    not something a re-save can do by accident.
    """
    errors: list[str] = []
    url = model = ""
    try:
        url = _validate_url(base_url_value)
    except SettingsError as exc:
        errors.append(str(exc))
    try:
        model = _validate_model(model_value)
    except SettingsError as exc:
        errors.append(str(exc))
    key: str | None = None
    if api_key_value is not None and _clean(api_key_value):
        try:
            key = _validate_key(api_key_value)
        except SettingsError as exc:
            errors.append(str(exc))
    if errors:
        raise SettingsError("\n".join(errors))

    # Cleared settings are stored empty rather than deleted: the settings table
    # holds value NOT NULL, and get_setting already reads an empty string as
    # "not configured".
    state.set_setting(SETTING_BASE_URL, url)
    state.set_setting(SETTING_MODEL, model)
    if clear_api_key:
        state.set_setting(SETTING_API_KEY, "")
    elif key is not None:
        state.set_setting(SETTING_API_KEY, key)


# --------------------------------------------------------------------------
# connectivity
# --------------------------------------------------------------------------
def check_connection(base_url_value: str = "", model_value: str = "",
                     api_key_value: str | None = None,
                     flavor_value: str | None = None) -> CheckResult:
    """Ask an endpoint for its model list and report what happened.

    Called with no arguments this tests the saved override. Called with the
    values from an unsaved form it tests those, which is the useful order: an
    operator should find out that an address is wrong before it becomes the
    address every extraction uses.

    api_key_value is a sentinel argument: None means "use the stored key", and
    a string - including an empty one - is used as given.

    A value the form did not send falls back to what is saved before it falls
    back to what is in force. In local mode those differ, and the in-force
    answer would be the local model's name - which is precisely the pairing
    that made the endpoint setting look configured while it was not.

    flavor_value follows the same sentinel rule as the key: None means "use the
    saved tickbox". It has to be here at all because the two dialects list
    their models at different addresses - /v1/models against an OpenAI-style
    server, /api/tags against Ollama's native API - so a test run in the wrong
    dialect answers 404 against a server that is working perfectly.
    """
    url = (_clean(base_url_value) or stored_base_url()
           or _local_base_url().value)
    try:
        url = _validate_url(url)
    except SettingsError as exc:
        return CheckResult(False, str(exc), url)

    want = (_clean(model_value) or stored_text_model()
            or _clean(env_str("TEXT_MODEL", "")))
    key = _api_key() if api_key_value is None else _clean(api_key_value)
    flavor = (stored_api_flavor() if flavor_value is None
              else _clean(flavor_value).lower())
    # The url above can fall back to the built-in runner when no endpoint is
    # saved, and that runner speaks only the OpenAI dialect. Testing it over
    # Ollama's native routes because a tickbox was left ticked answers 404
    # against a server that is working perfectly, and sends the operator
    # looking for a fault that is not there.
    if url.rstrip("/") == _local_base_url().value.rstrip("/"):
        flavor = FLAVOR_OPENAI

    # allow_override=False and an explicit url: the test must measure the
    # endpoint being tested, never quietly fall back to whatever is saved.
    runner = ModelRunner(url=url, api_key=key, allow_override=False,
                         timeout=CHECK_TIMEOUT, retries=1, flavor=flavor)
    try:
        models = runner.list_models()
    except ModelRunnerError as exc:
        # The client scrubs the key from its own error text; the first line is
        # what the callers of this pipeline show, so it is what is shown here.
        return CheckResult(False, str(exc).splitlines()[0], runner.url)
    except Exception as exc:                     # unexpected, still not fatal
        return CheckResult(False, f"the test failed: {exc}", runner.url)

    if not models:
        return CheckResult(
            True, f"reached {runner.url}, but it lists no models", runner.url,
            models, None if not want else False)
    if not want:
        return CheckResult(
            True, f"reached {runner.url}: {len(models)} model(s) available",
            runner.url, models, None)
    found = any(same_model(want, have) for have in models)
    if found:
        return CheckResult(True, f"reached {runner.url} and it serves {want}",
                           runner.url, models, True)
    return CheckResult(
        False,
        f"reached {runner.url}, but it does not serve {want} - it offers "
        f"{', '.join(models[:8])}" + (" ..." if len(models) > 8 else ""),
        runner.url, models, False)


# --------------------------------------------------------------------------
# the vision model
#
# Transcription is the one stage that hands a picture to a model, and it has
# always run on the local runner with a model named by VLM_MODEL in .env. That
# pinning was deliberate - a server chosen for its text model may serve no
# vision model at all - but it made the two settings move independently in the
# worst way: the operator pointed the text model at their own Ollama server,
# uploaded a scan, and was told VLM_MODEL was not set, naming a file they had
# never edited about a model they had never chosen.
#
# So the vision model is configured here, in the same place and in the same
# vocabulary as the text model, with one addition the text model has no use
# for. Where the endpoint can be asked what a model actually does, it is asked,
# because this is the one place in the pipeline where the wrong model does not
# announce itself. A text-only model given a page image returns fluent, ordered,
# entirely invented text, and that text is written to disk as the page's
# contents and read afterwards as evidence.
# --------------------------------------------------------------------------
def vision_scope() -> str:
    """Where transcription runs: the local runner, or the operator's endpoint.

    Anything that is not exactly the external marker reads as local, which is
    the same rule mode() follows and for a stronger reason. Local is what
    transcription has always done; an unset key, an older database and a word
    nothing here recognises must all leave page images on this machine rather
    than send them to a server on the strength of a value nobody can read.
    """
    return (VISION_EXTERNAL
            if _setting(SETTING_VISION_SCOPE) == VISION_EXTERNAL
            else VISION_LOCAL)


def is_vision_external() -> bool:
    """True when transcription should use the operator's own endpoint."""
    return vision_scope() == VISION_EXTERNAL


def set_vision_scope(value: str) -> None:
    """Store where vision runs. The only writer, and it validates its input.

    Refusing an unrecognised value rather than storing it, exactly as set_mode
    does: vision_scope() would read it back as local, and the operator would be
    told their scans were going to their server while they were not.
    """
    chosen = _clean(value).lower()
    if chosen not in VISION_SCOPES:
        raise SettingsError(
            f"the vision scope must be {VISION_LOCAL!r} or "
            f"{VISION_EXTERNAL!r} (got {value!r})")
    state.set_setting(SETTING_VISION_SCOPE, chosen)


def stored_vision_model() -> str:
    """The saved vision model name, whatever scope is in force."""
    return _setting(SETTING_VISION_MODEL)


def vision_model() -> Resolved:
    """Which vision model to ask, and where that name came from.

    The stored name applies under either scope, unlike the text model, whose
    stored name is read only in external mode. The difference is that the text
    model has an environment variable per mode to fall back on and vision does
    not: reading the stored name only when external would leave the settings
    page unable to name a model on the local runner at all, which is half of
    what this feature is for.

    VLM_MODEL remains the fallback for the local runner, so a machine that has
    been working from .env goes on working untouched. It is deliberately not a
    fallback for someone else's endpoint - that is the mistake this whole
    module exists to prevent, and it has already been made once here with
    TEXT_MODEL: the local model's name on a remote server produces an error
    about a model that server has never heard of.
    """
    stored = _setting(SETTING_VISION_MODEL)
    if stored:
        return Resolved(stored, "settings")
    if is_vision_external():
        return Resolved("", "unset")
    from_env = _clean(env_str("VLM_MODEL", ""))
    if from_env:
        return Resolved(from_env, "environment")
    return Resolved("", "unset")


def effective_vision_model() -> str:
    """The model name transcription should ask for; empty when none is set."""
    return vision_model().value


def vision_model_label() -> str:
    """How to name the source of the vision model in an error message.

    The label follows the value for the reason text_model_label() exists: an
    operator sent to .env for a name they chose on a page looks in the wrong
    file, and an operator sent to a page for a name that came from .env finds a
    field that is empty and correct.
    """
    source = vision_model().source
    if source == "settings":
        return "the vision model chosen on the settings page"
    if source == "environment":
        return "VLM_MODEL in .env"
    return "the vision model"


def vision_base_url() -> Resolved:
    """The endpoint transcription talks to, and where that address came from.

    External scope borrows the text model's saved endpoint rather than keeping
    a second one. An operator with two endpoints is not the case this was built
    for; an operator with one, who has already typed it in, is.
    """
    if not is_vision_external():
        return _local_base_url()
    stored = _setting(SETTING_BASE_URL)
    if stored:
        return Resolved(stored, "settings")
    # External with nothing saved is a misconfiguration rather than a fallback,
    # for the reason base_url() gives: naming the local address under a heading
    # that says otherwise is how an operator comes to believe a check passed.
    return Resolved("", "unset")


def vision_api_flavor() -> str:
    """The dialect transcription speaks: "openai" or "ollama".

    Local scope is the built-in Model Runner, which speaks the OpenAI dialect
    and nothing else, so the tickbox that belongs to the operator's endpoint
    must not follow vision back onto the local runner. External scope uses the
    saved dialect, because it is the saved endpoint being talked to.
    """
    return stored_api_flavor() if is_vision_external() else FLAVOR_OPENAI


def vision_can_be_verified() -> bool:
    """Whether this endpoint can be asked what a model is capable of.

    Only Ollama's native API answers that question. The page needs to know
    before it asks, so that it can say "not verified" as a property of the
    dialect rather than as the outcome of a check that appeared to fail.
    """
    return vision_api_flavor() == FLAVOR_OLLAMA


def vision_client_override() -> tuple[str, str, str]:
    """The endpoint, key and dialect for the client transcription builds.

    Local scope returns ("", "", "openai"), which is exactly the chain
    transcription has always followed - the environment, then the built-in
    runner, unix socket included. Nothing stored is consulted in that scope.

    External scope raises rather than returning empty when no endpoint is
    saved. An empty return would send page images to the local runner while the
    page said they were going elsewhere, and this is the stage where a quiet
    substitution does the most damage.
    """
    if not is_vision_external():
        return "", "", FLAVOR_OPENAI
    url = _setting(SETTING_BASE_URL)
    if not url:
        raise ModelRunnerError(
            "transcription is set to run on the external endpoint, but no "
            "endpoint address is saved - enter one on the settings page, or "
            "set transcription back to the local runner")
    return url, _api_key(), stored_api_flavor()


def vision_missing_message() -> str:
    """What to say when no vision model is configured anywhere.

    The first line stands alone, because the callers of this pipeline print
    only the first line - and it has to carry both halves of the news: where to
    fix it, and what it stops. An operator who reads "no vision model" without
    the second half has no way to tell whether their whole upload is broken or
    only the scans in it.
    """
    return (
        "No vision model is set, so pages that can only be read from an image "
        "cannot be transcribed - choose a vision model on the settings page.\n"
        "This affects scans and photographs only. A PDF whose text can be "
        "extracted never reaches this stage and is unaffected.\n"
        "VLM_MODEL in .env is still honoured for the local runner, if you "
        "would rather set it there.")


def save_vision_model(model: str | None) -> None:
    """Store the vision model name. None leaves it alone; "" clears it.

    The same empty-means-cleared rule the endpoint fields follow, with the same
    None sentinel for a field the form did not send at all.
    """
    if model is None:
        return
    state.set_setting(SETTING_VISION_MODEL, _validate_model(model))


def save_vision_config(model: str | None = None,
                       scope: str | None = None) -> None:
    """Store what the vision half of the form sent, in a safe order.

    The model is written before the scope for the reason main.py writes the
    endpoint before the mode: the scope must never be in force for the moment
    before the model it names has been stored. Either may be None, meaning the
    form did not send it and the saved value stands.
    """
    save_vision_model(model)
    if scope is not None:
        set_vision_scope(scope)


@dataclass(frozen=True)
class VisionConfig:
    """The effective vision configuration, safe to hand to a template."""

    model: Resolved
    scope: str = VISION_LOCAL
    stored_model: str = ""
    url: Resolved = Resolved("", "unset")
    api_flavor: str = FLAVOR_OPENAI

    @property
    def is_external(self) -> bool:
        return self.scope == VISION_EXTERNAL

    @property
    def can_be_verified(self) -> bool:
        """Whether this endpoint could be asked about a model's capabilities."""
        return self.api_flavor == FLAVOR_OLLAMA

    @property
    def is_set(self) -> bool:
        return bool(self.model.value)


def vision_config() -> VisionConfig:
    """Everything the settings page needs about vision, and nothing else."""
    return VisionConfig(model=vision_model(), scope=vision_scope(),
                        stored_model=stored_vision_model(),
                        url=vision_base_url(), api_flavor=vision_api_flavor())


def vision_config_as_dict() -> dict:
    """The effective vision configuration as plain data, for a JSON response."""
    cfg = vision_config()
    return {"scope": cfg.scope,
            "is_external": cfg.is_external,
            "can_be_verified": cfg.can_be_verified,
            "is_set": cfg.is_set,
            "api_flavor": cfg.api_flavor,
            "model": {"value": cfg.model.value, "source": cfg.model.source},
            "url": {"value": cfg.url.value, "source": cfg.url.source},
            "stored": {"model": cfg.stored_model}}


@dataclass
class VisionCheckResult:
    """The outcome of one vision check, in terms an operator can act on.

    vision is three-valued on purpose. True and False are answers from the
    endpoint; None means it was not able to be asked, which is neither a pass
    nor a failure and must not be rendered as either. ok says whether the
    configuration can be used at all, so it is True for an unverifiable model -
    transcription will run - and the page is expected to show the difference
    rather than a green tick.
    """

    ok: bool
    message: str
    url: str = ""
    model: str = ""
    vision: bool | None = None
    capabilities: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.vision is True

    def as_dict(self) -> dict:
        return {"ok": self.ok, "message": self.message, "url": self.url,
                "model": self.model, "vision": self.vision,
                "capabilities": self.capabilities, "models": self.models,
                "verified": self.verified}


def check_vision(model_value: str = "",
                 scope_value: str | None = None) -> VisionCheckResult:
    """Check that the chosen vision model exists and can actually see.

    Called with no arguments this tests what is saved. Called with the values
    from an unsaved form it tests those, which is the order that helps: an
    operator should learn that a model cannot read an image before a scan is
    transcribed by it.

    Two questions are asked, and they are different questions. The first -
    does this endpoint serve this model - can be asked of anything, and is the
    one the text model's check already asks. The second - does this model
    report vision - can only be asked of Ollama's native API, and where it
    cannot be asked this says so rather than passing quietly.
    """
    scope = (vision_scope() if scope_value is None
             else _clean(scope_value).lower())
    if scope not in VISION_SCOPES:
        return VisionCheckResult(
            False, f"the vision scope must be {VISION_LOCAL!r} or "
                   f"{VISION_EXTERNAL!r} (got {scope_value!r})")

    want = _clean(model_value) or stored_vision_model()
    if not want and scope == VISION_LOCAL:
        want = _clean(env_str("VLM_MODEL", ""))

    if scope == VISION_EXTERNAL:
        url, key, flavor = stored_base_url(), _api_key(), stored_api_flavor()
        if not url:
            return VisionCheckResult(
                False, "no endpoint address is saved, so there is nowhere to "
                       "run vision - enter one, or keep vision on the local "
                       "runner")
    else:
        # The local runner is reached exactly as transcription reaches it, the
        # unix socket included, so what is tested here is what will run.
        url, key, flavor = _local_base_url().value, "", FLAVOR_OPENAI

    try:
        url = _validate_url(url)
    except SettingsError as exc:
        return VisionCheckResult(False, str(exc), url)
    if not want:
        return VisionCheckResult(
            False, "no vision model is chosen, so scans and photographs "
                   "cannot be read", url)

    # allow_override=False with an explicit url, as the text check does: the
    # test must measure what it names. from_settings follows the scope so that
    # a remedy names the settings page rather than MODEL_URL.
    runner = ModelRunner(url=url, api_key=key, allow_override=False,
                         timeout=CHECK_TIMEOUT, retries=1, flavor=flavor,
                         from_settings=scope == VISION_EXTERNAL)
    try:
        models = runner.list_models()
    except ModelRunnerError as exc:
        return VisionCheckResult(False, str(exc).splitlines()[0], runner.url,
                                 want)
    except Exception as exc:                     # unexpected, still not fatal
        return VisionCheckResult(False, f"the test failed: {exc}", runner.url,
                                 want)

    if not models:
        return VisionCheckResult(
            False, f"reached {runner.url}, but it lists no models, so it "
                   f"cannot be confirmed to serve {want}", runner.url, want,
            models=models)
    if not any(same_model(want, have) for have in models):
        return VisionCheckResult(
            False,
            f"reached {runner.url}, but it does not serve {want} - it offers "
            f"{', '.join(models[:8])}" + (" ..." if len(models) > 8 else ""),
            runner.url, want, models=models)

    caps = runner.capabilities(want)
    listed = list(caps.values)
    if caps.vision is True:
        return VisionCheckResult(
            True, f"{runner.url} serves {want} and it reports vision - it can "
                  f"read page images", runner.url, want, True, listed, models)
    if caps.vision is False:
        # Known, and known to be wrong. This is the case worth the whole
        # exercise: the model would answer, and the answer would be invented.
        return VisionCheckResult(
            False,
            f"{runner.url} serves {want}, but it reports no vision capability "
            f"(it reports {', '.join(listed)}) - transcription will refuse it, "
            f"because a model that cannot see a page image writes a plausible "
            f"one instead. Choose a model that reports vision.",
            runner.url, want, False, listed, models)
    return VisionCheckResult(
        True,
        f"{runner.url} serves {want}, but whether it can read images could not "
        f"be verified: {caps.detail}. Transcription will run - check yourself "
        f"that this model reads images.",
        runner.url, want, None, listed, models)


def check_vision_connectivity(model_value: str = "",
                              scope_value: str | None = None) -> dict:
    """check_vision() as a plain dictionary, for the browser."""
    return check_vision(model_value, scope_value).as_dict()


# --------------------------------------------------------------------------
# the shape the page and its routes ask for
#
# The reader above answers in the module's own terms - base_url, text_model, a
# masked hint. The page and its routes speak of url, model and api_key, each
# carrying its own source so the operator can tell a saved override from an
# environment default. Rather than rename either side, this is the seam
# between them, and it is deliberately the only place the two vocabularies
# meet.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class KeyView:
    """A stored key as the page is allowed to see it: masked, never whole."""

    masked: str
    source: str
    is_set: bool


@dataclass(frozen=True)
class PageConfig:
    """What is in force, plus the mode and the values a form must render.

    url and model are the effective ones, so in local mode they describe the
    built-in runner. stored_url and stored_model are what would come back if
    the operator switched to external, which is what the input fields have to
    show - a field that empties itself when the mode changes reads as data
    loss.
    """

    url: Resolved
    model: Resolved
    api_key: KeyView
    mode: str = MODE_LOCAL
    stored_url: str = ""
    stored_model: str = ""
    api_flavor: str = FLAVOR_OPENAI
    stored_api_flavor: str = FLAVOR_OPENAI
    # Vision travels with the rest rather than beside it, so the page reads one
    # object. It is a whole configuration of its own - model, scope, endpoint,
    # dialect - because vision does not have to run where the text model runs.
    vision: VisionConfig | None = None

    @property
    def is_external(self) -> bool:
        return self.mode == MODE_EXTERNAL

    @property
    def is_ollama_api(self) -> bool:
        """Whether the dialect is in force, which is not the same as ticked.

        The tickbox renders from stored_api_flavor; this says whether it is
        doing anything, which in local mode it is not.
        """
        return self.api_flavor == FLAVOR_OLLAMA


def effective_config() -> PageConfig:
    """The effective text-model configuration, safe to render."""
    key = _api_key()
    return PageConfig(
        url=base_url(),
        model=text_model(),
        api_key=KeyView(masked=mask_key(key),
                        source="settings" if key else "unset",
                        is_set=bool(key)),
        mode=mode(),
        stored_url=stored_base_url(),
        stored_model=stored_text_model(),
        api_flavor=api_flavor(),
        stored_api_flavor=stored_api_flavor(),
        vision=vision_config())


def config_as_dict() -> dict:
    """The effective configuration as plain data, for a JSON response."""
    cfg = effective_config()
    return {"mode": cfg.mode,
            "is_external": cfg.is_external,
            "api_flavor": cfg.api_flavor,
            "is_ollama_api": cfg.is_ollama_api,
            "url": {"value": cfg.url.value, "source": cfg.url.source},
            "model": {"value": cfg.model.value, "source": cfg.model.source},
            "api_key": {"masked": cfg.api_key.masked,
                        "source": cfg.api_key.source,
                        "is_set": cfg.api_key.is_set},
            "stored": {"url": cfg.stored_url, "model": cfg.stored_model,
                       "api_flavor": cfg.stored_api_flavor},
            # Nested rather than flattened in: every key under "vision"
            # describes the vision model, and a reader that skips the block
            # cannot mistake one of them for a text-model setting.
            "vision": vision_config_as_dict()}


def save_config(url: str | None = None, model: str | None = None,
                api_key: str | None = None) -> None:
    """Store what the form sent.

    A field the form omitted entirely is left as it stands. That matters most
    for the key: the page cannot render it, so it posts one only when the
    operator typed a new one, and an omitted key must not be read as a request
    to delete the stored one. An explicit empty string is that request.

    "As it stands" means the saved value, not the effective one. In local mode
    the effective endpoint is the built-in runner, and reading that back would
    write the local address into the operator's endpoint field behind them.

    The mode is not written here. It has its own writer, set_mode(), because it
    is the one field with no text box and no empty-means-unchanged rule. The
    API flavour is the second such field and has its own writer for the same
    reason: set_api_flavor().
    """
    save(base_url_value=stored_base_url() if url is None else url,
         model_value=stored_text_model() if model is None else model,
         api_key_value=None if api_key in (None, "") else api_key,
         clear_api_key=api_key == "")


def check_connectivity() -> dict:
    """Test what is actually in force, as a plain dictionary for the browser."""
    return check_connection().as_dict()
