"""Immutable status domain values and privacy-safe metadata validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal

try:
    from .privacy import normalize_persistence_reason
except ImportError:  # Direct execution with scripts/ on sys.path.
    from privacy import normalize_persistence_reason


RunState = Literal["queued", "running", "retrying", "succeeded", "failed", "dead"]
EventLevel = Literal["info", "warning", "error"]
type JsonScalar = str | int | float | bool | None

_RUN_STATES = frozenset({"queued", "running", "retrying", "succeeded", "failed", "dead"})
_EVENT_LEVELS = frozenset({"info", "warning", "error"})

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
_ALLOWED_DETAIL_KEYS = frozenset(
    {"chars_saved", "changed_files", "retry_at", "elapsed_ms"}
)
_NONNEGATIVE_INTEGER_DETAIL_KEYS = frozenset(
    {"chars_saved", "changed_files", "elapsed_ms"}
)
_RETRY_AT_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)


def normalize_status_reason(
    value: object | None,
    env: Mapping[str, str],
) -> str | None:
    """Return a bounded, single-line, credential-redacted message or error."""
    if value is None:
        return None
    return normalize_persistence_reason(value, env)


def normalize_summary(
    value: object | None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Normalize, redact, and bound optional persisted summary text."""
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if not normalized:
        return None
    return normalize_persistence_reason(normalized, env or {})[:MAX_SUMMARY_CHARS]


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
        if not isinstance(key, str) or key not in _ALLOWED_DETAIL_KEYS:
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
        if key in _NONNEGATIVE_INTEGER_DETAIL_KEYS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"status detail {key!r} must be a nonnegative integer")
            normalized[key] = value
            continue
        if not isinstance(value, str) or _RETRY_AT_TIMESTAMP.fullmatch(value) is None:
            raise ValueError("status detail 'retry_at' must be a timezone-aware ISO-8601 timestamp")
        try:
            retry_at = datetime.fromisoformat(value)
            canonical_retry_at = retry_at.astimezone(UTC).isoformat(
                timespec="microseconds"
            )
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "status detail 'retry_at' must be a timezone-aware ISO-8601 timestamp"
            ) from error
        if retry_at.tzinfo is None or retry_at.utcoffset() is None:
            raise ValueError(
                "status detail 'retry_at' must be a timezone-aware ISO-8601 timestamp"
            )
        normalized[key] = canonical_retry_at
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
    redaction_env: InitVar[Mapping[str, str] | None] = None

    def __post_init__(self, redaction_env: Mapping[str, str] | None) -> None:
        if self.state not in _RUN_STATES:
            raise ValueError(f"invalid status run state: {self.state!r}")
        if self.phase not in ALLOWED_PHASES:
            raise ValueError(f"invalid status phase: {self.phase!r}")
        env = redaction_env or {}
        object.__setattr__(self, "summary", normalize_summary(self.summary, env))
        object.__setattr__(self, "error", normalize_status_reason(self.error, env))


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
    redaction_env: InitVar[Mapping[str, str] | None] = None

    def __post_init__(self, redaction_env: Mapping[str, str] | None) -> None:
        if self.phase not in ALLOWED_PHASES:
            raise ValueError(f"invalid status phase: {self.phase!r}")
        if self.level not in _EVENT_LEVELS:
            raise ValueError(f"invalid status event level: {self.level!r}")
        object.__setattr__(
            self,
            "message",
            normalize_status_reason(self.message, redaction_env or {}),
        )
        object.__setattr__(self, "details", normalize_details(self.details))
