"""Immutable status domain values and privacy-safe metadata validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal, TypeAlias

try:
    from .queue import normalize_persistence_reason
except ImportError:  # Direct execution with scripts/ on sys.path.
    from queue import normalize_persistence_reason  # type: ignore[attr-defined]


RunState = Literal["queued", "running", "retrying", "succeeded", "failed", "dead"]
EventLevel = Literal["info", "warning", "error"]
JsonScalar: TypeAlias = str | int | float | bool | None

ALLOWED_PHASES = frozenset(
    {
        "queued",
        "worker_claimed",
        "codex_started",
        "codex_succeeded",
        "codex_failed",
        "claude_started",
        "claude_succeeded",
        "claude_failed",
        "daily_log_write_started",
        "retry_wait",
        "recovery_pending",
        "succeeded",
        "failed",
        "dead",
        "reserved",
        "staging_started",
        "provider_started",
        "validation_started",
        "apply_started",
        "generation_recovered",
    }
)

MAX_SUMMARY_CHARS = 1_000
MAX_DETAIL_ITEMS = 32
MAX_DETAIL_STRING_CHARS = 1_000
_DETAIL_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FORBIDDEN_DETAIL_KEYS = frozenset(
    {"prompt", "transcript", "content", "output", "rendered_context"}
)


def normalize_status_reason(
    value: object | None,
    env: Mapping[str, str],
) -> str | None:
    """Return a bounded, single-line, credential-redacted message or error."""
    if value is None:
        return None
    return normalize_persistence_reason(value, env)


def normalize_summary(value: object | None) -> str | None:
    """Normalize optional summary text and cap its persisted size."""
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    return normalized[:MAX_SUMMARY_CHARS] or None


def normalize_details(
    details: Mapping[str, object] | None,
) -> Mapping[str, JsonScalar]:
    """Validate bounded operational metadata and return an immutable copy."""
    if details is None:
        return MappingProxyType({})
    if len(details) > MAX_DETAIL_ITEMS:
        raise ValueError(f"status details must contain at most {MAX_DETAIL_ITEMS} items")

    normalized: dict[str, JsonScalar] = {}
    for key, value in details.items():
        if not isinstance(key, str) or not _DETAIL_KEY.fullmatch(key):
            raise ValueError("status detail keys must be lowercase snake_case")
        if key in _FORBIDDEN_DETAIL_KEYS:
            raise ValueError(f"status detail key {key!r} is not permitted")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError(f"status detail {key!r} must be a scalar JSON value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"status detail {key!r} must be finite")
        if isinstance(value, str) and len(value) > MAX_DETAIL_STRING_CHARS:
            raise ValueError(
                f"status detail {key!r} must contain at most "
                f"{MAX_DETAIL_STRING_CHARS} characters"
            )
        normalized[key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class StatusRun:
    id: int
    job_id: int | None
    operation_key: str | None
    kind: str
    source_agent: str
    session_id: str
    project: str
    state: RunState
    phase: str
    summary: str | None
    error: str | None
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class StatusEvent:
    id: int
    run_id: int
    phase: str
    level: EventLevel
    provider: str | None
    attempt: int | None
    message: str | None
    details: Mapping[str, JsonScalar]
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", normalize_details(self.details))
