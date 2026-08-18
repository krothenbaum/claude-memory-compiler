"""Immutable status domain values and privacy-safe metadata validation."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal, get_args

try:
    from .privacy import normalize_persistence_reason
except ImportError:  # Direct execution with scripts/ on sys.path.
    from privacy import normalize_persistence_reason


RunState = Literal["queued", "running", "retrying", "succeeded", "failed", "dead"]
EventLevel = Literal["info", "warning", "error"]
type JsonScalar = str | int | float | bool | None

_RUN_STATES = frozenset(get_args(RunState))
_EVENT_LEVELS = frozenset(get_args(EventLevel))

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


def normalize_event_message(
    value: object | None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Normalize optional informational text without fabricating an error."""
    return normalize_summary(value, env)


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
        if self.attempt is not None and (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("status event attempt must be a positive integer")
        object.__setattr__(
            self,
            "message",
            normalize_event_message(self.message, redaction_env),
        )
        object.__setattr__(self, "details", normalize_details(self.details))


def _stored_time(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("status timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _loaded_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("persisted status timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def status_run_from_row(
    row: sqlite3.Row,
    *,
    redaction_env: Mapping[str, str] | None = None,
) -> StatusRun:
    """Build and validate an immutable run from a SQLite row."""
    started_at = _loaded_time(row["started_at"])
    updated_at = _loaded_time(row["updated_at"])
    if started_at is None or updated_at is None:
        raise ValueError("persisted status run is missing a required timestamp")
    return StatusRun(
        id=row["id"],
        job_id=row["job_id"],
        operation_key=row["operation_key"],
        kind=row["kind"],
        source_agent=row["source_agent"],
        session_id=row["session_id"],
        project=row["project"],
        state=row["state"],
        phase=row["phase"],
        summary=row["summary"],
        error=row["error"],
        started_at=started_at,
        updated_at=updated_at,
        completed_at=_loaded_time(row["completed_at"]),
        redaction_env=redaction_env,
    )


def status_event_from_row(
    row: sqlite3.Row,
    *,
    redaction_env: Mapping[str, str] | None = None,
) -> StatusEvent:
    """Build and validate an immutable event from a SQLite row."""
    created_at = _loaded_time(row["created_at"])
    if created_at is None:
        raise ValueError("persisted status event is missing its timestamp")
    details = json.loads(row["details_json"])
    if not isinstance(details, dict):
        raise TypeError("persisted status event details must be a JSON object")
    return StatusEvent(
        id=row["id"],
        run_id=row["run_id"],
        phase=row["phase"],
        level=row["level"],
        provider=row["provider"],
        attempt=row["attempt"],
        message=row["message"],
        details=details,
        created_at=created_at,
        redaction_env=redaction_env,
    )


def create_job_run_unlocked(
    connection: sqlite3.Connection,
    *,
    job_id: int,
    kind: str,
    source_agent: str,
    session_id: str,
    project: str,
    now: datetime,
    redaction_env: Mapping[str, str] | None = None,
) -> int:
    """Create a queued job run using the caller's existing transaction."""
    candidate = StatusRun(
        id=0,
        job_id=job_id,
        operation_key=None,
        kind=kind,
        source_agent=source_agent,
        session_id=session_id,
        project=project,
        state="queued",
        phase="queued",
        summary=None,
        error=None,
        started_at=now,
        updated_at=now,
        completed_at=None,
        redaction_env=redaction_env,
    )
    cursor = connection.execute(
        """
        INSERT INTO status_runs (
            job_id, kind, source_agent, session_id, project, state, phase,
            summary, error, started_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate.job_id,
            candidate.kind,
            candidate.source_agent,
            candidate.session_id,
            candidate.project,
            candidate.state,
            candidate.phase,
            candidate.summary,
            candidate.error,
            _stored_time(candidate.started_at),
            _stored_time(candidate.updated_at),
            None,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("created status run has no identifier")
    return cursor.lastrowid


def create_operation_run_unlocked(
    connection: sqlite3.Connection,
    *,
    operation_key: str,
    kind: str,
    source_agent: str,
    session_id: str,
    project: str,
    phase: str,
    now: datetime,
    redaction_env: Mapping[str, str] | None = None,
) -> int:
    """Create a queued non-job run using the caller's transaction."""
    if not operation_key.strip():
        raise ValueError("operation_key must not be empty")
    candidate = StatusRun(
        id=0,
        job_id=None,
        operation_key=operation_key,
        kind=kind,
        source_agent=source_agent,
        session_id=session_id,
        project=project,
        state="queued",
        phase=phase,
        summary=None,
        error=None,
        started_at=now,
        updated_at=now,
        completed_at=None,
        redaction_env=redaction_env,
    )
    cursor = connection.execute(
        """
        INSERT INTO status_runs (
            operation_key, kind, source_agent, session_id, project, state, phase,
            summary, error, started_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate.operation_key,
            candidate.kind,
            candidate.source_agent,
            candidate.session_id,
            candidate.project,
            candidate.state,
            candidate.phase,
            candidate.summary,
            candidate.error,
            _stored_time(candidate.started_at),
            _stored_time(candidate.updated_at),
            None,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("created status run has no identifier")
    return cursor.lastrowid


def append_event_unlocked(
    connection: sqlite3.Connection,
    run_id: int,
    phase: str,
    *,
    now: datetime,
    level: EventLevel = "info",
    provider: str | None = None,
    attempt: int | None = None,
    message: str | None = None,
    details: Mapping[str, JsonScalar] | None = None,
    redaction_env: Mapping[str, str] | None = None,
) -> int:
    """Append a validated event using the caller's existing transaction."""
    candidate = StatusEvent(
        id=0,
        run_id=run_id,
        phase=phase,
        level=level,
        provider=provider,
        attempt=attempt,
        message=message,
        details={} if details is None else details,
        created_at=now,
        redaction_env=redaction_env,
    )
    cursor = connection.execute(
        """
        INSERT INTO status_events (
            run_id, phase, level, provider, attempt, message, details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate.run_id,
            candidate.phase,
            candidate.level,
            candidate.provider,
            candidate.attempt,
            candidate.message,
            json.dumps(dict(candidate.details), sort_keys=True, separators=(",", ":")),
            _stored_time(candidate.created_at),
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("created status event has no identifier")
    return cursor.lastrowid


def transition_run_unlocked(
    connection: sqlite3.Connection,
    run_id: int,
    state: RunState,
    phase: str,
    *,
    now: datetime,
    summary: str | None = None,
    error: str | None = None,
    completed_at: datetime | None = None,
    level: EventLevel = "info",
    provider: str | None = None,
    attempt: int | None = None,
    message: str | None = None,
    details: Mapping[str, JsonScalar] | None = None,
    redaction_env: Mapping[str, str] | None = None,
) -> None:
    """Update a run and append its matching event in the caller's transaction."""
    row = connection.execute(
        "SELECT * FROM status_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    existing = status_run_from_row(row, redaction_env=redaction_env)
    candidate = StatusRun(
        id=existing.id,
        job_id=existing.job_id,
        operation_key=existing.operation_key,
        kind=existing.kind,
        source_agent=existing.source_agent,
        session_id=existing.session_id,
        project=existing.project,
        state=state,
        phase=phase,
        summary=summary,
        error=error,
        started_at=existing.started_at,
        updated_at=now,
        completed_at=completed_at,
        redaction_env=redaction_env,
    )
    connection.execute(
        """
        UPDATE status_runs
        SET state = ?, phase = ?, summary = ?, error = ?, updated_at = ?,
            completed_at = ?
        WHERE id = ?
        """,
        (
            candidate.state,
            candidate.phase,
            candidate.summary,
            candidate.error,
            _stored_time(candidate.updated_at),
            (
                _stored_time(candidate.completed_at)
                if candidate.completed_at is not None
                else None
            ),
            candidate.id,
        ),
    )
    append_event_unlocked(
        connection,
        run_id,
        phase,
        now=now,
        level=level,
        provider=provider,
        attempt=attempt,
        message=message,
        details=details,
        redaction_env=redaction_env,
    )
