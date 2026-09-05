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

from pipeline import chat, embed, graph, paths, report, runner, state  # noqa: E402
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
        "c": _counts(), "objective": report.get_objective(), "previous": previous,
        "chat_model": os.environ.get("TEXT_MODEL", "").strip(),
    })


@app.post("/api/objective")
def api_objective(payload: dict):
    """The objective is entered once and kept, so the button is one click."""
    report.set_objective(str(payload.get("objective") or ""))
    return JSONResponse({"saved": True,
                         "allegations": len(report.split_allegations(report.get_objective()))})


@app.post("/api/report")
def api_report(payload: dict):
    objective = payload.get("objective")

    def events():
        try:
            for kind, data in report.generate(objective):
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
        except Exception as exc:
            log.error("report stream failed: %s", exc)
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"

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
