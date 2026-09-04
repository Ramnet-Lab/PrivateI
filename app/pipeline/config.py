"""Environment reading with fail-fast behaviour and useful error text."""
from __future__ import annotations

import os


class ConfigError(RuntimeError):
    pass


def env_str(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise ConfigError(f"{name} is not set")
    return val


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer") from exc


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number") from exc


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}
