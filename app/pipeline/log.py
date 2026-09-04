"""Uniform logging.  Console for the operator, JSONL on disk for the record."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

from . import paths


class _JsonlHandler(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "service": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                row["exc"] = self.format(record).split("\n", 1)[-1]
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # logging must never take the run down
            pass


def get_logger(service: str) -> logging.Logger:
    log = logging.getLogger(service)
    if log.handlers:
        return log
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s  %(message)s",
                                           datefmt="%H:%M:%S"))
    log.addHandler(console)
    log.addHandler(_JsonlHandler(paths.LOGS / f"{service}.jsonl"))
    log.propagate = False
    return log


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
