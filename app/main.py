#!/usr/bin/env python3
"""Upload documents, process them, look at the graph.

Everything runs in this one container: reading files, OCR, calling the models
on the host, and loading the result into Neo4j.  Uploads are queued and handled
one at a time by a background worker.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import zipfile
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_DIR = Path(os.environ.get("APP_DIR", "/app"))
sys.path.insert(0, str(APP_DIR))

from pipeline import (chat, consensus, embed, graph, llm_settings,  # noqa: E402
                      model_client, paths, report, runner, state)
from pipeline.log import get_logger, utcnow               # noqa: E402

log = get_logger("app")

app = FastAPI(title="Document Analysis", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024


@app.on_event("startup")
def startup() -> None:
    paths.ensure_tree()
    state.init_db()
    resumed = runner.requeue_unfinished()
    if resumed:
        log.info("resumed %d document(s) that were mid-process at shutdown", resumed)
    log.info("ready on 8080")


# --- pages -----------------------------------------------------------------

def _counts() -> dict:
    def n(sql: str, params=()) -> int:
        row = state.query_one(sql, params)
        return int(row["n"]) if row else 0
    return {
        "documents": n("SELECT COUNT(*) AS n FROM documents"),
        "processing": n("SELECT COUNT(*) AS n FROM documents "
                        "WHERE status IN ('queued','processing')"),
        "pages": n("SELECT COUNT(*) AS n FROM pages"),
        "assertions": n("SELECT COUNT(*) AS n FROM triples"),
        "entities": n("SELECT COUNT(*) AS n FROM entities WHERE merged_into IS NULL"),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    docs = state.query("SELECT * FROM documents ORDER BY uploaded_at DESC")
    return templates.TemplateResponse(request, "index.html", {
        "docs": docs, "c": _counts(), "accepted": sorted(paths.SUPPORTED),
        "graph_up": graph.available(),
    })


@app.get("/graph", response_class=HTMLResponse)
def graph_page(request: Request):
    return templates.TemplateResponse(request, "graph.html", {
        "c": _counts(), "graph_up": graph.available(),
    })


@app.get("/timeline", response_class=HTMLResponse)
def timeline_page(request: Request):
    rows = graph.timeline() if graph.available() else []
    return templates.TemplateResponse(request, "timeline.html", {
        "rows": rows, "c": _counts(),
    })


@app.get("/documents/{doc_id}", response_class=HTMLResponse)
def document_page(request: Request, doc_id: str):
    doc = state.query_one("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    pages = state.query(
        "SELECT * FROM pages WHERE doc_id=? ORDER BY page_num", (doc_id,))
    texts = {}
    for page in pages:
        if page["text_path"]:
            found = paths.under_root(page["text_path"])
            texts[page["page_num"]] = found.read_text(encoding="utf-8") if found else ""
    assertions = state.query(
        "SELECT * FROM triples WHERE doc_id=? ORDER BY page_num", (doc_id,))
    return templates.TemplateResponse(request, "document.html", {
        "doc": doc, "pages": pages, "texts": texts, "assertions": assertions,
        "c": _counts(),
    })


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {
        "c": _counts(), "index": embed.stats(),
        # The name of the text model has to come from the resolver, not from
        # the environment: when the operator has chosen an external endpoint on
        # the settings page, the environment still holds the prepackaged name
        # and this page would name a model the pipeline is not using. The
        # embedding model is read from the environment on purpose - embeddings
        # never follow the override and always run on the local runner.
        "chat_model": llm_settings.effective_text_model(),
        "embed_model": os.environ.get("EMBED_MODEL", "").strip(),
    })


@app.post("/api/chat")
async def api_chat(payload: dict):
    """Server-sent events: sources first, then the answer as it is written."""
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="ask a question")
    history = payload.get("history") or []

    def events():
        try:
            for kind, data in chat.answer(question, history):
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        except Exception as exc:            # never leave the page hanging
            log.error("chat stream failed: %s", exc)
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/reindex")
def api_reindex():
    """Index everything already processed - run after setting EMBED_MODEL."""
    def work():
        try:
            embed.backfill()
        except Exception as exc:
            log.error("reindex failed: %s", exc)

    threading.Thread(target=work, name="reindex", daemon=True).start()
    return JSONResponse({"started": True})


@app.get("/api/index")
def api_index():
    return JSONResponse(embed.stats())


@app.get("/report", response_class=HTMLResponse)
def report_page(request: Request):
    previous = state.query(
        "SELECT report_id, created_at, model, documents, assertions FROM reports "
        "ORDER BY created_at DESC LIMIT 20")
    return templates.TemplateResponse(request, "report.html", {
        "c": _counts(), "goal": report.get_goal(),
        "allegations": report.get_allegations(), "previous": previous,
        # Same resolver as the chat page: this value gates the run button, so
        # reading the environment here would refuse to write a report that a
        # configured external endpoint could write, or offer to write one with
        # a model name the pipeline will not use.
        "chat_model": llm_settings.effective_text_model(),
        # The seed is shown beside each stored report because the pipeline no
        # longer samples at temperature zero: without it a report cannot be
        # regenerated, and the operator would have no way to tell which of two
        # differing reports was which.
        "seeds": consensus.seeds(), "max_runs": consensus.MAX_RUNS,
    })


@app.post("/api/objective")
def api_objective(payload: dict):
    """Goal and allegations arrive as separate fields - nothing is parsed."""
    allegations = payload.get("allegations") or []
    if not isinstance(allegations, list):
        raise HTTPException(status_code=400, detail="allegations must be a list")
    report.set_objective(str(payload.get("goal") or ""),
                         [str(a) for a in allegations])
    return JSONResponse({"saved": True, "allegations": len(report.get_allegations())})


def _sse(kind: str, data) -> str:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"


def _runs_requested(value) -> int:
    """How many independent runs of the report to generate and compare.

    Bounded on both sides. Each run is a full report - several model passes per
    allegation on CPU - so an unbounded number here is an unbounded amount of
    machine time started by one click, and a request for zero runs is a request
    for nothing rather than for the default.
    """
    if value is None or value == "":
        return 1
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise HTTPException(status_code=400, detail="runs must be a whole number")
    try:
        runs = int(str(value).strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="runs must be a whole number")
    if not 1 <= runs <= consensus.MAX_RUNS:
        raise HTTPException(status_code=400,
                            detail=f"runs must be between 1 and {consensus.MAX_RUNS}")
    return runs


def _report_runs(goal, allegations, runs: int, allow_incomplete: bool):
    """Stream one report per run, then the comparison between them.

    Every run draws its own seed. Sampling alone would not make the second run
    an independent draw - the same seed at the same temperature reproduces the
    same text - so without a fresh seed each time the comparison below would be
    a document compared with a copy of itself.

    A run that fails stops the queue: the refusals are deterministic, so a
    second attempt under the same conditions would fail the same way. Whatever
    completed before the failure is still compared, because throwing away four
    good runs because the fifth timed out helps nobody. The runs that never
    started stay in the list and say so, so the denominator of every agreement
    ratio remains the number of runs the operator asked for - a comparison that
    quietly shrinks its own denominator reports better agreement the more runs
    it loses.
    """
    members = [{"n": n, "seed": None, "report_id": None,
                "error": "not run - an earlier run failed", "allegations": None}
               for n in range(1, runs + 1)]
    for member in members:
        number, seed = member["n"], model_client.random_seed()
        member["seed"], member["error"] = seed, None
        yield _sse("run", {"n": number, "of": runs, "seed": seed})
        if runs > 1:
            yield _sse("token", f"\n\n===== Run {number} of {runs} "
                                f"(seed {seed}) =====\n\n")

        for kind, data in report.generate(goal, allegations, seed=seed,
                                          allow_incomplete=allow_incomplete):
            if kind == "done":
                member["report_id"] = (data or {}).get("report_id")
                continue
            if kind == "error":
                member["error"] = str(data)
            elif kind == "status" and runs > 1:
                data = f"Run {number} of {runs}: {data}"
            yield _sse(kind, data)

        if member["report_id"]:
            consensus.record_seed(member["report_id"], seed)
            yield _sse("run-done", {"n": number, "of": runs, "seed": seed,
                                    "report_id": member["report_id"]})
        if member["error"]:
            log.error("run %d of %d failed: %s", number, runs, member["error"])
            break

    if runs == 1:
        if members[0]["report_id"]:
            yield _sse("done", {"report_id": members[0]["report_id"], "runs": 1})
        return

    yield _sse("status", "Comparing runs")
    for member in members:
        if not member["report_id"]:
            member["error"] = member["error"] or "the run produced no report"
            continue
        row = state.query_one("SELECT body FROM reports WHERE report_id=?",
                              (member["report_id"],))
        try:
            if not row:
                raise consensus.UnreadableRun("the report was not stored")
            member["allegations"] = consensus.read_run(row["body"])
        except consensus.UnreadableRun as exc:
            # Loud, and counted against agreement rather than dropped. A run
            # whose dispositions could not be parsed did not agree with the
            # others, and silently comparing only the runs that parsed would
            # report a parsing failure as consensus.
            member["error"] = f"could not be read: {exc}"
            log.error("run %d (%s) %s", member["n"], member["report_id"],
                      member["error"])
            yield _sse("error", f"Run {member['n']} {member['error']}")

    summary = consensus.compare(members)
    listing = [{k: m[k] for k in ("n", "seed", "report_id", "error")}
               for m in members]
    created = utcnow()
    # The model recorded against a stored report must be the model that wrote
    # it. The environment name is the prepackaged one, which is the wrong
    # answer whenever the settings page points the text model elsewhere.
    model = llm_settings.effective_text_model()
    body = consensus.render(summary, listing, created, model)
    yield _sse("token", "\n\n" + body)

    done_ids = [m["report_id"] for m in members if m["report_id"]]
    report_id = consensus.save(done_ids, body, created) if done_ids else None
    yield _sse("consensus", {"summary": summary, "runs": listing,
                             "report_id": report_id})
    if report_id:
        yield _sse("done", {"report_id": report_id, "runs": runs})


@app.post("/api/report")
def api_report(payload: dict):
    goal = payload.get("goal")
    allegations = payload.get("allegations")
    if allegations is not None and not isinstance(allegations, list):
        raise HTTPException(status_code=400, detail="allegations must be a list")
    runs = _runs_requested(payload.get("runs"))
    # The corpus-integrity refusal is deliberate, so the override is deliberate
    # too: it is off unless the operator asked for it on this request. Without a
    # route into the running app the refusal cannot be cleared at all, and a
    # worker that died mid-stage leaves a document that only a restart or a hand
    # edit of the database can get past.
    allow_incomplete = bool(payload.get("allow_incomplete"))

    def events():
        try:
            yield from _report_runs(goal, allegations, runs, allow_incomplete)
        except Exception as exc:
            log.error("report stream failed: %s", exc)
            yield _sse("error", str(exc))

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/reports/{report_id}", response_class=PlainTextResponse)
def report_markdown(report_id: str):
    row = state.query_one("SELECT body FROM reports WHERE report_id=?", (report_id,))
    if not row:
        raise HTTPException(status_code=404, detail="report not found")
    return PlainTextResponse(row["body"],
                             media_type="text/markdown; charset=utf-8",
                             headers={
        "Content-Disposition": f'attachment; filename="report_{report_id}.md"'})


# --- text model endpoint ---------------------------------------------------
#
# The whole page is a thin shell over pipeline.llm_settings, which owns where
# the text model actually points and is the only place that reads the stored
# key. This module deliberately never handles the secret except to hand a
# newly typed one straight to the writer: effective_config() returns the key
# already masked, so there is no full key here to leak into a log line, a
# template or an error message. Each field it returns carries the source it
# came from as well as its value, because an operator who cannot tell a saved
# override from an environment default cannot tell whether their change took
# effect.
#
# The mode is the choice that matters most on the page - the prepackaged local
# model, or the operator's own endpoint - so it travels with every save. It has
# its own writer in that module rather than riding along with the text fields,
# because it is the one setting with no box to leave empty.
#
# The vision model rides on the same route and the same page, and it is a
# separate setting throughout: its own name, its own choice of where it runs,
# and its own check. It has to be separate because it is a separate model. An
# operator who moved the text model to their own server found transcription
# still failing against an environment variable they had never set, because the
# two had silently been one decision made in two places.


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {
        # effective_config() carries the mode, what each field resolves to
        # under it, and the saved endpoint values the form has to render even
        # when the mode is not using them. The page needs no second source.
        "c": _counts(), "cfg": llm_settings.effective_config(),
        # Vision is a second configuration rather than more fields on the
        # first, because it is a second model with its own place to run. The
        # page renders the two as two sections for the same reason.
        "vision": llm_settings.vision_config(),
        # Named on the page so the operator knows exactly which file on disk
        # now holds the API key they just typed.
        "db_path": str(paths.STATE_DB),
    })


@app.post("/api/settings")
def api_settings(payload: dict):
    """Save the mode and the endpoint fields. An empty string clears a field.

    api_key is the exception to "empty clears": the page cannot show the stored
    key, so it sends the field only when the operator actually typed a new one.
    A missing api_key therefore means "leave it alone", and only an explicit
    empty string clears it - otherwise saving a new URL would silently discard
    the key.

    The page omits url and model entirely while the local model is chosen,
    which is what keeps a switch back to local from emptying the endpoint the
    operator means to return to. api_flavor - the dialect that endpoint speaks
    - follows the same rule: it describes the external endpoint, so the page
    posts it only when that endpoint is the choice being saved, and an omitted
    value leaves the stored dialect exactly as it stands.

    vision_model and vision_scope live on the same rule and are the reason it
    is worth stating twice. The page saves the vision section on its own
    button, so a save from the text section posts neither of them and the
    vision configuration is not touched; a save from the vision section posts
    only those two and leaves the endpoint, its key and its dialect alone.
    """
    url = payload.get("url")
    model = payload.get("model")
    api_key = payload.get("api_key")
    mode = payload.get("mode")
    api_flavor = payload.get("api_flavor")
    vision_model = payload.get("vision_model")
    vision_scope = payload.get("vision_scope")
    for name, value in (("url", url), ("model", model), ("api_key", api_key),
                        ("mode", mode), ("api_flavor", api_flavor),
                        ("vision_model", vision_model),
                        ("vision_scope", vision_scope)):
        if value is not None and not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"{name} must be text")
    # The mode is checked before anything is written. set_mode() would refuse
    # an unrecognised value anyway, but refusing it here means a save that is
    # going to fail does not first store half of itself.
    if mode is not None and mode not in llm_settings.MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {', '.join(llm_settings.MODES)}")
    # The dialect is checked in the same breath and for the same reason. It
    # decides the shape of every extraction request, and the wrong shape does
    # not fail loudly: an Ollama server answers an OpenAI-style call, drops the
    # context size it was given and returns an empty completion.
    if api_flavor is not None and api_flavor not in llm_settings.FLAVORS:
        raise HTTPException(
            status_code=400,
            detail=f"api_flavor must be one of {', '.join(llm_settings.FLAVORS)}")
    # And the vision scope in the same breath, before anything is written. Its
    # two values happen to be spelled the same as the text mode's, which is
    # exactly why it is checked against its own tuple: the day a third text mode
    # appears, it must not become a third place to run vision by accident.
    if (vision_scope is not None
            and vision_scope not in llm_settings.VISION_SCOPES):
        raise HTTPException(
            status_code=400,
            detail="vision_scope must be one of "
                   f"{', '.join(llm_settings.VISION_SCOPES)}")
    try:
        # The endpoint values are stored first and the mode is switched after,
        # so external mode is never in force for the moment before the address
        # it points at has been written.
        llm_settings.save_config(url=url, model=model, api_key=api_key)
        # The dialect is written before the mode is switched, for the same
        # reason the address is: external mode must never be in force for the
        # moment before the way to speak to that address has been stored.
        if api_flavor is not None:
            llm_settings.set_api_flavor(api_flavor)
        # Vision goes through its own writer, which stores the model name before
        # it switches the scope on the same reasoning the two lines above
        # follow. Both halves take None to mean the form did not send them, so
        # a save from either section of the page leaves the other alone.
        llm_settings.save_vision_config(model=vision_model, scope=vision_scope)
        if mode is not None:
            llm_settings.set_mode(mode)
    except llm_settings.SettingsError as exc:
        # What the operator typed cannot be stored as given. This is their
        # error to fix and the message names the field, so it belongs on the
        # page rather than in a 500 that reads as a fault in the application.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info("model settings updated: mode=%s dialect=%s "
             "vision=%s on the %s runner",
             llm_settings.mode(), llm_settings.api_flavor(),
             llm_settings.effective_vision_model() or "not set",
             llm_settings.vision_scope())
    return JSONResponse({"saved": True,
                         "config": llm_settings.config_as_dict(),
                         "vision": llm_settings.vision_config_as_dict()})


@app.post("/api/settings/test")
def api_settings_test():
    """Ask the saved endpoint what it offers, and report what happened.

    The check runs against the saved configuration rather than against anything
    posted here, so the key never travels back and forth over this route. That
    is also what lets an address be checked while the local model is still the
    one in force, which is the order an operator wants: find out that an
    endpoint serves the model before every extraction depends on it.

    A failure is returned as a failure with its own message: reporting it as
    anything softer would leave an operator believing a remote model is in use
    when it is not.
    """
    outcome = llm_settings.check_connectivity()
    # The outcome names the dialect it was measured over. A green line saying
    # only "reachable" reads as proof of whichever API the operator has in
    # mind, and those two are precisely what this setting lets them confuse:
    # an Ollama server answers a model list on both routes while honouring the
    # context size on only one. The check reports no dialect of its own, so it
    # is named here, and it is the saved one rather than the one in force
    # because check_connectivity() tests the saved configuration and reaches
    # for the saved tickbox to do it.
    outcome.setdefault("api_flavor", llm_settings.stored_api_flavor())
    return JSONResponse(outcome)


@app.post("/api/settings/vision/test")
def api_settings_vision_test():
    """Ask whether the saved vision model exists and can actually read images.

    This is two questions rather than one. Whether the endpoint serves the model
    can be asked of anything; whether the model can see can only be asked of an
    Ollama server, over /api/show. The answer to the second is therefore
    three-valued - yes, no, or not askable here - and the third is not a pass.
    The page renders all three differently, so a model that could not be
    verified is never shown as a model that was.

    It checks what is saved rather than what is posted, exactly as the text
    check above does and for the same reason: nothing about this route needs a
    request body, and the stored key never travels back and forth over it.
    """
    outcome = llm_settings.check_vision_connectivity()
    # The scope and the dialect name what the answer was measured against. A
    # line that said only "cannot be verified" would read as a fact about the
    # model, when it is a fact about the endpoint being asked - and an operator
    # who switched vision to their Ollama server would have no way to see that
    # the check had gone to the local runner instead.
    outcome.setdefault("scope", llm_settings.vision_scope())
    outcome.setdefault("api_flavor", llm_settings.vision_api_flavor())
    outcome.setdefault("can_be_verified", llm_settings.vision_can_be_verified())
    return JSONResponse(outcome)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    state.query_one("SELECT 1 AS n")
    return "ok"


# --- upload ----------------------------------------------------------------

@app.post("/upload")
async def upload(files: list[UploadFile]):
    accepted, skipped = [], []

    for item in files:
        name = paths.safe_filename(item.filename or "upload")
        suffix = Path(name).suffix.lower()
        if suffix not in paths.SUPPORTED:
            skipped.append({"filename": name, "reason": f"{suffix or 'no extension'} "
                            f"is not a supported type"})
            continue

        # Stream to a temp file, hashing as it goes, so a large PDF never has to
        # sit in memory in full.
        paths.RAW.mkdir(parents=True, exist_ok=True)
        tmp = paths.RAW / f".incoming_{os.getpid()}_{name}"
        digest = hashlib.sha256()
        size = 0
        try:
            with tmp.open("wb") as out:
                while chunk := await item.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise ValueError(f"larger than the "
                                         f"{MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
                    digest.update(chunk)
                    out.write(chunk)
        except ValueError as exc:
            tmp.unlink(missing_ok=True)
            skipped.append({"filename": name, "reason": str(exc)})
            continue
        finally:
            await item.close()

        sha = digest.hexdigest()
        existing = state.query_one("SELECT doc_id FROM documents WHERE sha256=?", (sha,))
        if existing:
            tmp.unlink(missing_ok=True)
            skipped.append({"filename": name,
                            "reason": "already uploaded (identical contents)"})
            continue

        doc_id = paths.doc_id_for(name, sha)
        stored_name = f"{doc_id}{Path(name).suffix.lower()}"
        os.replace(tmp, paths.RAW / stored_name)

        suffix_kind = ("pdf" if suffix in paths.SUPPORTED_PDF else
                       "word" if suffix in paths.SUPPORTED_DOC else "image")
        with state.tx() as conn:
            conn.execute(
                """INSERT INTO documents (doc_id, filename, sha256, size_bytes,
                       media_type, uploaded_at, status)
                   VALUES (?,?,?,?,?,?, 'queued')""",
                (doc_id, stored_name, sha, size, suffix_kind, utcnow()))
        runner.enqueue(doc_id)
        accepted.append({"doc_id": doc_id, "filename": name})
        log.info("accepted %s as %s (%d bytes)", name, doc_id, size)

    return JSONResponse({"accepted": accepted, "skipped": skipped,
                         "queued": runner.queue_depth()})


@app.post("/documents/{doc_id}/delete")
def delete_document(doc_id: str):
    doc = state.query_one("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")

    if graph.available():
        with graph.driver().session() as session:
            session.run("""MATCH ()-[r]->() WHERE r.source_doc = $doc DELETE r""",
                        doc=doc_id)
            # Entities left with no assertions at all are no longer evidence of
            # anything, so they go too.
            session.run("""MATCH (e:Entity) WHERE NOT (e)-[]-() DELETE e""")

    with state.tx() as conn:
        conn.execute("DELETE FROM triples WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
    # Entities are derived; a deleted document's people must not survive it.
    from pipeline import extract as _extract
    _extract.rebuild_entities()

    for directory in (paths.PAGES / doc_id, paths.TEXT / doc_id):
        shutil.rmtree(directory, ignore_errors=True)
    (paths.RAW / doc["filename"]).unlink(missing_ok=True)
    log.info("deleted %s", doc_id)
    return JSONResponse({"deleted": doc_id})


def _reset_document(doc_id: str) -> None:
    """Return a document to the state it was in the moment after upload.

    Retry resumes where processing stopped; re-ingest deliberately does not.
    Page images, read text, facts, passages and graph edges are all derived
    from the original file, so all of them are discarded and rebuilt - that is
    what makes a prompt change or an OCR fix actually reach existing documents
    instead of only new ones. The uploaded file itself is never touched.
    """
    if graph.available():
        with graph.driver().session() as session:
            session.run("MATCH ()-[r]->() WHERE r.source_doc = $doc DELETE r",
                        doc=doc_id)
            session.run("MATCH (e:Entity) WHERE NOT (e)-[]-() DELETE e")

    with state.tx() as conn:
        conn.execute("DELETE FROM triples WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        conn.execute("UPDATE documents SET page_count=0, error=NULL WHERE doc_id=?",
                     (doc_id,))

    # Drop the rendered pages and read text so ingestion and OCR run again
    # rather than skipping work that already exists on disk.
    for directory in (paths.PAGES / doc_id, paths.TEXT / doc_id):
        shutil.rmtree(directory, ignore_errors=True)


@app.post("/documents/{doc_id}/reingest")
def reingest_document(doc_id: str):
    if not state.query_one("SELECT 1 AS n FROM documents WHERE doc_id=?", (doc_id,)):
        raise HTTPException(status_code=404, detail="document not found")
    # A run of THIS document is superseded by this request and is abandoned
    # before anything else happens; a run of any other document is not, and is
    # left alone. The cancel comes before the reset because the abandoned run
    # goes on writing pages and assertions until it lets go, and rows deleted
    # while it is still writing come back as the leavings of a run nobody
    # wanted. It comes before the enqueue for a stronger reason: cancelling
    # after the replacement is queued could abort the replacement itself.
    with runner.paused():
        aborted = runner.cancel_inflight(doc_id)
        _reset_document(doc_id)
        runner.enqueue(doc_id, force=True)
    log.info("re-ingesting %s from the original file%s", doc_id,
             " (the run in flight was abandoned)" if aborted else "")
    return JSONResponse({"reingesting": doc_id, "aborted": aborted})


@app.post("/api/reingest-all")
def reingest_all():
    """Rebuild every document from its original file.

    Entities are rebuilt from surviving triples afterwards by the extraction
    stage; clearing them here as well keeps a stale roster from being offered
    to chat and reports during the rebuild.
    """
    # Every document is about to be rebuilt, so whatever the worker is inside
    # of is superseded whichever document it belongs to - including the model
    # call it is blocked in, which is the whole point: that call was made with
    # the settings this request exists to replace, and until it returns nothing
    # else can start. Everything here happens in one order for one reason. The
    # abort is first, before a row is reset, so the run being abandoned is no
    # longer writing to rows this is about to delete. It is also before
    # anything is queued, because an abort issued afterwards could land on one
    # of the new runs instead and leave a document abandoned that the operator
    # had just asked for.
    with runner.paused():
        aborted = runner.cancel_inflight()
        docs = state.query("SELECT doc_id FROM documents ORDER BY uploaded_at")
        for row in docs:
            _reset_document(row["doc_id"])
        with state.tx() as conn:
            conn.execute("DELETE FROM entities")
        for row in docs:
            runner.enqueue(row["doc_id"], force=True)
    log.info("re-ingesting all %d document(s)%s", len(docs),
             f" (abandoned the run of {aborted})" if aborted else "")
    return JSONResponse({"reingesting": len(docs), "aborted": aborted})


@app.post("/documents/{doc_id}/retry")
def retry_document(doc_id: str):
    if not state.query_one("SELECT 1 AS n FROM documents WHERE doc_id=?", (doc_id,)):
        raise HTTPException(status_code=404, detail="document not found")
    runner.enqueue(doc_id)
    return JSONResponse({"queued": doc_id})


# --- purge -----------------------------------------------------------------
#
# Destroying the case at closure is a designed operation, not a convenience:
# the products transfer and the data tree is destroyed per the records
# disposition schedule. It is also the one action here that cannot be undone,
# so the phrase below is required in the request itself. A boolean would be
# satisfied by a stray click or by a request replayed out of a log; a phrase
# that names the scope cannot be, and a phrase captured for one scope cannot
# stand in for a wider one.
# Longer than a re-ingest waits: a re-ingest overlapping a dying run
# redoes work, a purge overlapping one reports a destruction that did
# not happen.
PURGE_CANCEL_GRACE = 30.0

PURGE_PHRASES = {
    "documents": "DELETE ALL DOCUMENTS",
    "reports": "DELETE ALL REPORTS",
    "all": "DELETE EVERYTHING",
}


def _clear_directory(directory: Path) -> int:
    """Empty a directory, keeping the directory itself, and count what went.

    The directory stays because the tree is built once at startup and every
    writer below here assumes its own directory exists; removing it would
    leave the next upload failing on a missing path. An entry is only counted
    once it is actually gone, so the number returned is what was destroyed
    rather than what was attempted.
    """
    if not directory.exists():
        return 0
    removed = 0
    for child in sorted(directory.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        if not child.exists():
            removed += 1
    return removed


def _purge_originals(filenames: set[str]) -> tuple[int, list[str]]:
    """Delete the uploaded originals from 01_raw; name anything left behind.

    01_raw keeps no manifest, index or other record of its own. Every file in
    it was written by the upload route and named for the document it became,
    and the only other thing that can appear there is a ".incoming_" partial
    left by an upload that died before its file could be renamed into place.
    The manifest this application speaks of is a block written into the body
    of a report, not a file on disk, so there is no record here to preserve.

    That is why the rule is what it is: a file the documents table names, a
    partial, or a file carrying one of the extensions the upload route accepts
    is case content and is destroyed. Anything else was put there by a person
    rather than by this application, and it is kept and named in the response
    instead of being destroyed on a guess - an operator can then delete it
    deliberately, which is the only way anything should leave this directory.
    """
    if not paths.RAW.exists():
        return 0, []
    removed, kept = 0, []
    for child in sorted(paths.RAW.iterdir()):
        case_content = (child.name in filenames
                        or child.name.startswith(".incoming_")
                        or child.suffix.lower() in paths.SUPPORTED)
        if not case_content:
            kept.append(child.name)
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        if not child.exists():
            removed += 1
    return removed, kept


@app.get("/api/purge")
def api_purge_options():
    """The phrase each scope requires, and what that scope would destroy.

    The page needs both, and must not hold its own copy of either: a
    confirmation phrase written out in two places is a phrase that can come to
    disagree with itself, and a warning that names counts of its own is a
    warning that can name the wrong ones. graph_up is here because a purge of
    documents is refused while the graph database is unreachable, and the
    operator should learn that before typing the phrase rather than after.
    """
    counts = _counts()
    reports = state.query_one("SELECT COUNT(*) AS n FROM reports")
    return JSONResponse({
        "phrases": PURGE_PHRASES,
        "counts": {
            "documents": counts["documents"],
            "pages": counts["pages"],
            "assertions": counts["assertions"],
            "entities": counts["entities"],
            "reports": int(reports["n"]) if reports else 0,
        },
        "graph_up": graph.available(),
    })


@app.post("/api/purge")
def api_purge(payload: dict):
    """Destroy every document, every report, or both, and say what went.

    Scopes are kept apart on purpose. An operator clearing the documents to
    start a case again still wants the reports written so far, and an operator
    clearing drafts of a report has not asked to lose the corpus they were
    drawn from. "all" is offered because closure needs it, but it is a third
    choice with a phrase of its own rather than a side effect of either of the
    other two.

    The settings table survives every scope. The endpoint configuration and
    the investigation objective are not case material, and an operator
    clearing documents does not expect to lose the address of their model.
    """
    scope = payload.get("scope")
    confirm = payload.get("confirm")
    # Both checks come before anything is read, let alone deleted. A malformed
    # request destroys nothing at all.
    if not isinstance(scope, str) or scope not in PURGE_PHRASES:
        raise HTTPException(
            status_code=400,
            detail=f"scope must be one of {', '.join(PURGE_PHRASES)}")
    if not isinstance(confirm, str):
        raise HTTPException(status_code=400, detail="confirm must be text")
    required = PURGE_PHRASES[scope]
    # Compared exactly, allowing only for whitespace around what was typed.
    # Accepting a different case would make the phrase a formality, and the
    # point of it is that the operator has to write out what they are about to
    # destroy before it happens.
    if confirm.strip() != required:
        raise HTTPException(
            status_code=400,
            detail=f'to purge {scope} you must type "{required}" exactly; '
                   f"nothing has been deleted")

    wants_documents = scope in ("documents", "all")
    wants_reports = scope in ("reports", "all")

    # A purge of documents needs the graph database, and is refused without
    # it. The alternative would be to clear SQLite and the disk and leave the
    # Neo4j entities and relationships standing, and those are not inert:
    # report generation reads the timeline out of the graph and the chat page
    # reads entity detail from it, so the next report would be written partly
    # from the leavings of the case that was supposed to have been destroyed.
    # Refusing leaves the operator with everything they had, which is a state
    # they can act on; a half-purge is not.
    if wants_documents and not graph.available():
        raise HTTPException(
            status_code=503,
            detail="the graph database is not reachable, and a purge that "
                   "left its entities and relationships behind would leave "
                   "them to be read into the next report; nothing has been "
                   "deleted - start the graph database and purge again")

    rows: dict[str, int] = {}
    files: dict[str, int] = {}
    graph_deleted: dict[str, int] = {}
    kept_in_raw: list[str] = []

    # The same wedge the re-ingest routes were built around. The worker is one
    # thread that spends most of its life blocked inside a model call, and it
    # goes on writing pages and assertions until it lets go, so deleting
    # underneath it leaves rows that reappear after the purge has reported
    # itself complete. The gate is closed for the whole of this so the worker
    # cannot start the next queued document into a tree being wiped, and the
    # abort comes first so that the run already in flight has been told to
    # stop before a single row is deleted.
    with runner.paused():
        # Only a purge of documents cancels: the worker never writes a report,
        # so aborting the run it is inside to clear the reports table would
        # abandon a document nobody asked to abandon.
        # A run that has not let go is still writing pages and assertions, and
        # deleting underneath it puts rows back after the purge has reported
        # success. Waiting longer here than a re-ingest does is deliberate: a
        # re-ingest overlapping a dying run merely redoes work, while a purge
        # overlapping one tells the operator material was destroyed when it
        # was not.
        aborted = (runner.cancel_inflight(grace=PURGE_CANCEL_GRACE)
                   if wants_documents else None)
        if wants_documents and runner.inflight() is not None:
            raise HTTPException(
                status_code=409,
                detail="a document is still being processed and would write "
                       "into what this deletes; wait for it to finish and "
                       "try again")

        if wants_documents:
            # Read before the rows go, because the originals on disk are named
            # by the rows and there is nothing else that knows their names.
            filenames = {row["filename"] for row in
                         state.query("SELECT filename FROM documents")}

            # Every node, not the per-document pattern of relationships then
            # orphans: the whole graph is derived from assertions that are
            # about to cease to exist, and sweeping it entirely also takes any
            # entity an earlier single-document delete left stranded. The
            # constraints and indexes are schema rather than case material and
            # stay. The counters are read from the statement itself so that
            # what is reported is what the database actually deleted.
            with graph.driver().session() as session:
                counters = session.run("MATCH (n) DETACH DELETE n").consume().counters
                graph_deleted = {"nodes": counters.nodes_deleted,
                                 "relationships": counters.relationships_deleted}

            with state.tx() as conn:
                rows["triples"] = conn.execute("DELETE FROM triples").rowcount
                rows["pages"] = conn.execute("DELETE FROM pages").rowcount
                rows["chunks"] = conn.execute("DELETE FROM chunks").rowcount
                rows["documents"] = conn.execute("DELETE FROM documents").rowcount
                # Entities are derived from assertions that no longer exist.
                # rebuild_entities() would arrive at the same empty roster,
                # but only by reading a table this transaction has just
                # emptied; deleting them outright says so directly and cannot
                # leave a stale name behind if that reading ever changes.
                rows["entities"] = conn.execute("DELETE FROM entities").rowcount

            # The passage vectors live in the chunks rows just deleted, and
            # the search matrix is a cached copy of them held in memory. It
            # rebuilds itself when the chunk count changes, so this is belt
            # and braces - but a purge that left chat answering out of a
            # matrix built from destroyed documents is exactly the failure
            # this endpoint exists to prevent.
            embed._invalidate()

            files["02_pages"] = _clear_directory(paths.PAGES)
            files["03_text"] = _clear_directory(paths.TEXT)
            files["01_raw"], kept_in_raw = _purge_originals(filenames)

            # Documents still sitting in the work queue are left there. The
            # worker checks for the row before it reads or writes anything and
            # skips a document that has been deleted, so a queue entry that
            # outlives its document brings nothing back.

        if wants_reports:
            with state.tx() as conn:
                rows["reports"] = conn.execute("DELETE FROM reports").rowcount
                # The seed each report was drawn with is kept in the settings
                # store rather than on the report row, so a report deleted
                # here would otherwise leave its seed behind forever. These
                # rows are the only part of that table that is case material:
                # they are per-report metadata, keyed by report id, and they
                # go with the reports. Matched by prefix rather than with LIKE
                # because the key contains an underscore, which LIKE reads as
                # a wildcard - the configuration keys could not match it in
                # practice, but a delete against the settings table is not the
                # place to rely on that.
                rows["report_seeds"] = conn.execute(
                    "DELETE FROM settings WHERE substr(key, 1, ?) = ?",
                    (len(consensus.SEED_KEY), consensus.SEED_KEY)).rowcount

            # Nothing in this application writes a report to disk - a report
            # is stored in the reports table and rendered on demand by
            # /reports/{report_id} - but 05_products is where the finished
            # products are put, and a finished product is a report. So the
            # directory is emptied with the table. It is swept rather than
            # matched by name because nothing here wrote those files and there
            # is no record of what they were called.
            files["05_products"] = _clear_directory(paths.PRODUCTS)

    log.warning("purged %s: rows %s, graph %s, files %s%s", scope,
                rows or "none", graph_deleted or "not touched", files or "none",
                f", abandoned the run of {aborted}" if aborted else "")
    return JSONResponse({"purged": True, "scope": scope, "rows": rows,
                         "graph": graph_deleted, "files": files,
                         "kept_in_01_raw": kept_in_raw, "aborted": aborted})


# --- data for the pages ----------------------------------------------------

@app.get("/api/documents")
def api_documents():
    rows = state.query(
        """SELECT doc_id, filename, media_type, page_count, status, stage,
                  progress, error, uploaded_at FROM documents
           ORDER BY uploaded_at DESC""")
    return JSONResponse({
        "documents": [dict(r) for r in rows],
        "counts": _counts(),
        "queued": runner.queue_depth(),
    })


@app.get("/api/graph")
def api_graph():
    if not graph.available():
        return JSONResponse({"nodes": [], "edges": [],
                             "error": "the graph database is not reachable"})
    return JSONResponse(graph.snapshot())


@app.get("/api/entity/{entity_id:path}")
def api_entity(entity_id: str):
    if not graph.available():
        raise HTTPException(status_code=503, detail="graph database not reachable")
    detail = graph.entity_detail(entity_id)
    if not detail:
        raise HTTPException(status_code=404, detail="entity not found")
    return JSONResponse(detail)


@app.get("/image/{doc_id}/{page_num}")
def page_image(doc_id: str, page_num: int):
    row = state.query_one(
        "SELECT image_path FROM pages WHERE doc_id=? AND page_num=?", (doc_id, page_num))
    if not row or not row["image_path"]:
        raise HTTPException(status_code=404, detail="no image for that page")
    # The path comes from the database, but it is still resolved and checked:
    # a traversal here would serve arbitrary files from the tree.
    found = paths.under_root(row["image_path"])
    if found is None:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(found, media_type="image/png")


def _transcript_text(doc_id: str) -> tuple[str, str]:
    """One document's pages as plain text, with a header saying where it came
    from and how each page was read. Returns (filename, text)."""
    doc = state.query_one("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    pages = state.query(
        "SELECT * FROM pages WHERE doc_id=? ORDER BY page_num", (doc_id,))

    lines = [
        f"# {doc['filename']}",
        f"# {doc['page_count']} page(s), uploaded {doc['uploaded_at']}",
        f"# sha256 {doc['sha256']}",
        "#",
        "# Text below was produced by OCR or a vision model and may contain",
        "# errors. The page images remain the authority.",
        "",
    ]
    for page in pages:
        how = page["text_source"] or "not read"
        if page["model"]:
            how = f"{how} ({page['model']})"
        if page["ocr_conf"] is not None:
            how = f"{how}, OCR confidence {page['ocr_conf']:.0f}"
        lines.append(f"----- page {page['page_num']} [{how}] -----")
        body = ""
        if page["text_path"]:
            found = paths.under_root(page["text_path"])
            if found:
                body = found.read_text(encoding="utf-8").rstrip()
        lines.append(body or "(no text was read from this page)")
        lines.append("")

    stem = Path(doc["filename"]).stem or doc_id
    return f"{stem}.txt", "\n".join(lines)


@app.get("/documents/{doc_id}/transcript.txt", response_class=PlainTextResponse)
def download_transcript(doc_id: str):
    name, text = _transcript_text(doc_id)
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8",
                             headers={
        "Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/transcripts.zip")
def download_all_transcripts():
    """Every document's transcript in one zip, for taking the text elsewhere."""
    rows = state.query("SELECT doc_id FROM documents ORDER BY uploaded_at")
    if not rows:
        raise HTTPException(status_code=404, detail="nothing has been uploaded yet")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        used: set[str] = set()
        for row in rows:
            name, text = _transcript_text(row["doc_id"])
            # Two uploads can share a filename; keep both rather than silently
            # dropping one.
            candidate, n = name, 2
            while candidate in used:
                candidate = f"{Path(name).stem} ({n}).txt"
                n += 1
            used.add(candidate)
            archive.writestr(candidate, text)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/zip", headers={
        "Content-Disposition": 'attachment; filename="transcripts.zip"'})


@app.get("/api/text/{doc_id}/{page_num}", response_class=PlainTextResponse)
def page_text(doc_id: str, page_num: int):
    row = state.query_one(
        "SELECT text_path FROM pages WHERE doc_id=? AND page_num=?", (doc_id, page_num))
    if not row or not row["text_path"]:
        raise HTTPException(status_code=404, detail="no text for that page")
    found = paths.under_root(row["text_path"])
    if found is None:
        raise HTTPException(status_code=404, detail="text not found")
    return PlainTextResponse(found.read_text(encoding="utf-8"),
                             media_type="text/plain; charset=utf-8",
                             headers={
        "Content-Disposition":
            f'attachment; filename="{doc_id}_page{page_num}.txt"'})
