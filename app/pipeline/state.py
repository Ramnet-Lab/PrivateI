"""SQLite state: what has been uploaded, how far it got, and what came out.

Not WAL.  The file sits on a bind mount shared with the Docker VM, and WAL
coordinates through a shared-memory index that is not coherent across that
boundary - a writer on one side leaves a reader on the other reporting
"database disk image is malformed" against a file that is actually intact.
TRUNCATE journalling coordinates through file locks alone.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from . import paths
from .log import utcnow

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    size_bytes  INTEGER NOT NULL,
    media_type  TEXT NOT NULL,
    page_count  INTEGER NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'queued',
    stage       TEXT,
    progress    TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    doc_id        TEXT NOT NULL,
    page_num      INTEGER NOT NULL,
    image_path    TEXT,
    ocr_conf      REAL,
    ocr_words     INTEGER,
    route         TEXT,
    text_path     TEXT,
    text_source   TEXT,
    model         TEXT,
    error         TEXT,
    PRIMARY KEY (doc_id, page_num)
);

CREATE TABLE IF NOT EXISTS triples (
    triple_id    TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    page_num     INTEGER NOT NULL,
    subject_type TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    predicate    TEXT NOT NULL,
    object_type  TEXT NOT NULL,
    object_name  TEXT NOT NULL,
    event_date   TEXT,
    quote        TEXT NOT NULL,
    model        TEXT,
    created_at   TEXT NOT NULL,
    loaded_at    TEXT
);
CREATE INDEX IF NOT EXISTS triples_doc ON triples(doc_id, page_num);

-- Small key/value store: the CDI objective is kept here so it persists.
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Generated reports, kept so one can be re-read without regenerating it.
CREATE TABLE IF NOT EXISTS reports (
    report_id  TEXT PRIMARY KEY,
    objective  TEXT NOT NULL,
    body       TEXT NOT NULL,
    model      TEXT,
    documents  INTEGER,
    assertions INTEGER,
    created_at TEXT NOT NULL
);

-- Page text split into passages, with an embedding each, for the chat page.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    page_num   INTEGER NOT NULL,
    ord        INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB,
    dims       INTEGER,
    model      TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS entities (
    entity_id      TEXT PRIMARY KEY,
    entity_type    TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    merged_into    TEXT,
    mention_count  INTEGER NOT NULL DEFAULT 0,
    first_seen     TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    paths.STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(paths.STATE_DB), timeout=30,
                           isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA journal_mode = TRUNCATE")
    _local.conn = conn
    return conn


def init_db() -> None:
    # executescript() commits any open transaction, so it cannot run inside tx().
    connect().executescript(SCHEMA)


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return connect().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    return connect().execute(sql, tuple(params)).fetchone()


def get_setting(key: str, default: str = "") -> str:
    row = query_one("SELECT value FROM settings WHERE key=?", (key,))
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, value, utcnow()))


def set_status(doc_id: str, status: str, *, stage: str | None = None,
               progress: str | None = None, error: str | None = None) -> None:
    with tx() as conn:
        conn.execute(
            """UPDATE documents SET status=?, stage=COALESCE(?, stage),
                   progress=?, error=?,
                   started_at = CASE WHEN started_at IS NULL AND ?='processing'
                                     THEN ? ELSE started_at END,
                   finished_at = CASE WHEN ? IN ('done','failed') THEN ? ELSE finished_at END
               WHERE doc_id=?""",
            (status, stage, progress, error, status, utcnow(), status, utcnow(), doc_id))
