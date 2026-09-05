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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from . import state
from .config import env_str
from .model_client import DEFAULT_URL, ModelRunner, ModelRunnerError, same_model

# The setting keys, named once. The web layer imports these rather than
# spelling the strings again, because a typo in a key name reads as "no
# override configured" and would fail silently.
SETTING_BASE_URL = "llm_base_url"
SETTING_MODEL = "llm_model"
SETTING_API_KEY = "llm_api_key"
SETTING_MODE = "llm_mode"

# The two modes, named once for the same reason as the keys. Local is the
# behaviour that existed before the settings page, and it is what an unset or
# unrecognised value means, so a database written by an older build - or by a
# future one that learns a third mode - degrades to the safe reading.
MODE_LOCAL = "local"
MODE_EXTERNAL = "external"
MODES = (MODE_LOCAL, MODE_EXTERNAL)

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
                     api_key_value: str | None = None) -> CheckResult:
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

    # allow_override=False and an explicit url: the test must measure the
    # endpoint being tested, never quietly fall back to whatever is saved.
    runner = ModelRunner(url=url, api_key=key, allow_override=False,
                         timeout=CHECK_TIMEOUT, retries=1)
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

    @property
    def is_external(self) -> bool:
        return self.mode == MODE_EXTERNAL


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
        stored_model=stored_text_model())


def config_as_dict() -> dict:
    """The effective configuration as plain data, for a JSON response."""
    cfg = effective_config()
    return {"mode": cfg.mode,
            "is_external": cfg.is_external,
            "url": {"value": cfg.url.value, "source": cfg.url.source},
            "model": {"value": cfg.model.value, "source": cfg.model.source},
            "api_key": {"masked": cfg.api_key.masked,
                        "source": cfg.api_key.source,
                        "is_set": cfg.api_key.is_set},
            "stored": {"url": cfg.stored_url, "model": cfg.stored_model}}


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
    is the one field with no text box and no empty-means-unchanged rule.
    """
    save(base_url_value=stored_base_url() if url is None else url,
         model_value=stored_text_model() if model is None else model,
         api_key_value=None if api_key in (None, "") else api_key,
         clear_api_key=api_key == "")


def check_connectivity() -> dict:
    """Test what is actually in force, as a plain dictionary for the browser."""
    return check_connection().as_dict()
