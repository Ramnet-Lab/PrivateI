"""Vision-model transcription for pages OCR could not read confidently."""
from __future__ import annotations

import time

from . import llm_settings, paths, state
from .ingest import write_text
from .log import get_logger
from .model_client import (ModelMissing, ModelNotSet, Ollama, OllamaError,
                           default_options, thinking_enabled)
from .prompts_transcribe import build as build_prompt

log = get_logger("transcribe")


def _client() -> Ollama:
    """The same endpoint the text model uses.

    Vision has no configuration of its own. An operator who points this
    deployment at their own server means all of it, and a second address to
    keep in step was one more thing to get wrong: the text model moved, the
    vision one did not, and transcription failed naming a variable nobody had
    set. A bare client resolves the operator's endpoint, key and dialect when
    one is configured, and the built-in runner with its socket when it is not.
    """
    return Ollama()


def _require_vision(client: Ollama, model: str, on_progress) -> None:
    """Refuse a model known not to see; say so when that cannot be known.

    This is the one stage where the wrong model does not announce itself. A
    text-only model handed a page image does not error and does not return
    nothing: it returns fluent, ordered, plausible text that was never on the
    page, which is written to disk as that page's contents and read afterwards
    as evidence. Ollama will say which models can see, so it is asked.

    Where it cannot be asked - the OpenAI dialect has no route that answers -
    the run goes ahead, because refusing every unverifiable endpoint would
    refuse the local runner, which is the default and has always worked. What
    it must not do is pass silently: the operator is told here, and the
    settings page says the same thing where they chose the model.
    """
    capabilities = client.capabilities(model)
    if capabilities.vision is False:
        raise ModelMissing(
            f"{model} reports no vision capability at {client.url}, so it "
            f"cannot read a page image - choose a model that reports vision, "
            f"on the settings page.\n"
            f"{client.url} reports {model} as: "
            f"{', '.join(capabilities.values)}.\n"
            f"This is refused rather than attempted because a model that "
            f"cannot see the page does not fail on one - it writes a plausible "
            f"page instead, and that text is read afterwards as evidence.")
    if capabilities.vision is None:
        log.warning("cannot verify that %s reads images: %s", model,
                    capabilities.detail)
        on_progress(f"{model} could not be checked for image support - "
                    f"transcribing anyway")
        return
    log.info("%s reports vision (%s)", model, ", ".join(capabilities.values))


def run(doc_id: str, on_progress) -> int:
    rows = state.query(
        """SELECT doc_id, page_num, image_path FROM pages
           WHERE doc_id=? AND route='vlm' AND text_path IS NULL
           ORDER BY page_num""", (doc_id,))
    if not rows:
        return 0

    # Transcription is a VISION request and does not follow the text endpoint.
    # A server chosen for its text model may serve no vision model at all, and
    # page images are the rawest case material there is, so where this runs is
    # its own setting with its own default: the local runner, which is what
    # this stage has always used.
    client = _client()
    # The name is resolved before the endpoint is asked anything, so that "no
    # vision model is set" is answered in its own words. require_model's
    # version of that message names .env or the settings page by where the
    # endpoint came from, and the answer here depends on neither: it is the
    # settings page in both scopes, and what it stops is scans and photographs
    # rather than the document.
    # The same model the rest of the pipeline uses. Whether it can read an
    # image is asked of the endpoint below rather than configured here.
    model = llm_settings.effective_text_model()
    if not model:
        raise ModelNotSet(
            "No model is set, so page images cannot be read. Choose one on "
            "the settings page. This stops scans and photographs only - a "
            "document whose text extracts directly never reaches this stage.")
    model = client.require_model(model, "the model on the settings page")
    _require_vision(client, model, on_progress)
    options = default_options("VLM_TEMPERATURE", "VLM_NUM_CTX",
                              "TRANSCRIBE_NUM_PREDICT", 2000)
    system, user, version = build_prompt()

    done = 0
    for idx, row in enumerate(rows, 1):
        on_progress(f"transcribing page {idx}/{len(rows)} with {model}")
        image = paths.under_root(row["image_path"])
        if image is None:
            continue
        started = time.time()
        try:
            data = client.generate(model, user, images=[image], system=system,
                                   options=options, think=thinking_enabled())
            text = (data.get("response") or "").strip()
            if not text:
                raise OllamaError("the model returned an empty transcription")
        except Exception as exc:
            # One bad page must not sink the document.
            log.error("%s p%s: %s", doc_id, row["page_num"], exc)
            with state.tx() as conn:
                conn.execute("UPDATE pages SET error=? WHERE doc_id=? AND page_num=?",
                             (str(exc)[:400], doc_id, row["page_num"]))
            continue

        txt = write_text(doc_id, row["page_num"], text)
        with state.tx() as conn:
            conn.execute(
                """UPDATE pages SET text_path=?, text_source='vlm', model=?, error=NULL
                   WHERE doc_id=? AND page_num=?""",
                (paths.rel(txt), model, doc_id, row["page_num"]))
        done += 1
        log.info("%s p%s: %d chars in %.1fs (%s, prompt %s)", doc_id, row["page_num"],
                 len(text), time.time() - started, model, version)
    return done
