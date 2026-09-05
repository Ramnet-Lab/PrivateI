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

from pipeline import (chat, consensus, embed, graph, model_client, paths,  # noqa: E402
                      report, runner, state)
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
        "chat_model": os.environ.get("TEXT_MODEL", "").strip(),
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
        "chat_model": os.environ.get("TEXT_MODEL", "").strip(),
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
    model = os.environ.get("TEXT_MODEL", "").strip()
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
    _reset_document(doc_id)
    runner.enqueue(doc_id, force=True)
    log.info("re-ingesting %s from the original file", doc_id)
    return JSONResponse({"reingesting": doc_id})


@app.post("/api/reingest-all")
def reingest_all():
    """Rebuild every document from its original file.

    Entities are rebuilt from surviving triples afterwards by the extraction
    stage; clearing them here as well keeps a stale roster from being offered
    to chat and reports during the rebuild.
    """
    docs = state.query("SELECT doc_id FROM documents ORDER BY uploaded_at")
    for row in docs:
        _reset_document(row["doc_id"])
    with state.tx() as conn:
        conn.execute("DELETE FROM entities")
    for row in docs:
        runner.enqueue(row["doc_id"], force=True)
    log.info("re-ingesting all %d document(s)", len(docs))
    return JSONResponse({"reingesting": len(docs)})


@app.post("/documents/{doc_id}/retry")
def retry_document(doc_id: str):
    if not state.query_one("SELECT 1 AS n FROM documents WHERE doc_id=?", (doc_id,)):
        raise HTTPException(status_code=404, detail="document not found")
    runner.enqueue(doc_id)
    return JSONResponse({"queued": doc_id})


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
