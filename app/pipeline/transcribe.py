"""Vision-model transcription for pages OCR could not read confidently."""
from __future__ import annotations

import time

from . import paths, state
from .config import env_str
from .ingest import write_text
from .log import get_logger
from .model_client import (Ollama, OllamaError, default_options,
                            thinking_enabled)
from .prompts_transcribe import build as build_prompt

log = get_logger("transcribe")


def run(doc_id: str, on_progress) -> int:
    rows = state.query(
        """SELECT doc_id, page_num, image_path FROM pages
           WHERE doc_id=? AND route='vlm' AND text_path IS NULL
           ORDER BY page_num""", (doc_id,))
    if not rows:
        return 0

    # Transcription is a VISION request and must not follow the text
    # endpoint. A server chosen for its text model may serve no vision model
    # at all, and page images are the rawest case material there is. Passing
    # allow_override=False keeps this on the local runner, socket included,
    # whatever the text model is pointed at.
    client = Ollama(allow_override=False)
    model = client.require_model(env_str("VLM_MODEL", ""), "VLM_MODEL")
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
