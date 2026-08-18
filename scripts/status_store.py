"""Immutable status domain values and privacy-safe metadata validation."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast, get_args

try:
    from .privacy import normalize_persistence_reason
    from .utils import (
        atomic_write_private_file,
        inspect_secure_read_file,
        read_private_bounded_file,
        secure_private_state_lock,
    )
except ImportError:  # Direct execution with scripts/ on sys.path.
    from privacy import normalize_persistence_reason
    from utils import (
        atomic_write_private_file,
        inspect_secure_read_file,
        read_private_bounded_file,
        secure_private_state_lock,
    )


RunState = Literal["queued", "running", "retrying", "succeeded", "failed", "dead"]
EventLevel = Literal["info", "warning", "error"]
ProviderName = Literal["codex", "claude"]
RunKind = Literal["capture", "compile", "query_file", "connections", "semantic_lint"]
SourceAgent = Literal["claude", "codex", "system"]
type JsonScalar = str | int | float | bool | None

_RUN_STATES = frozenset(get_args(RunState))
_EVENT_LEVELS = frozenset(get_args(EventLevel))
_PROVIDER_NAMES = frozenset(get_args(ProviderName))
_RUN_KINDS = frozenset(get_args(RunKind))
_SOURCE_AGENTS = frozenset(get_args(SourceAgent))

_TERMINAL_STATES = frozenset({"succeeded", "failed", "dead"})
_OPERATION_TRANSITIONS: Mapping[RunState, frozenset[RunState]] = {
    "queued": frozenset({"running", "failed", "dead"}),
    "running": frozenset({"retrying", "succeeded", "failed", "dead"}),
    "retrying": frozenset({"running", "failed", "dead"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "dead": frozenset(),
}
_OPERATION_PHASES: Mapping[RunState, frozenset[str]] = {
    "queued": frozenset({"queued", "reserved"}),
    "running": frozenset(
        {
            "staging_started",
            "provider_started",
            "validation_started",
            "apply_started",
            "generation_recovered",
        }
    ),
    "retrying": frozenset({"retry_wait", "recovery_pending"}),
    "succeeded": frozenset({"succeeded"}),
    "failed": frozenset({"failed"}),
    "dead": frozenset({"dead"}),
}

MAX_OPERATION_KEY_CHARS = 512
MAX_OPERATION_IDENTITY_CHARS = 256

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


def _validate_identity_text(
    field: str,
    value: str,
    *,
    max_chars: int,
    allow_empty: bool,
    redaction_env: Mapping[str, str],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    if not value:
        if allow_empty:
            return value
        raise ValueError(f"{field} must not be empty")
    if value != value.strip() or len(value) > max_chars:
        raise ValueError(f"{field} must be canonical and at most {max_chars} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} must not contain control characters")
    if normalize_persistence_reason(value, redaction_env) != value:
        raise ValueError(f"{field} must not contain secrets or noncanonical whitespace")
    return value


def validate_operation_identity(
    operation_key: str,
    kind: str,
    source_agent: str,
    session_id: str,
    project: str,
    *,
    redaction_env: Mapping[str, str] | None = None,
) -> tuple[str, RunKind, SourceAgent, str, str]:
    """Validate bounded, canonical identity fields before operation persistence."""
    env = redaction_env or {}
    key = _validate_identity_text(
        "operation_key",
        operation_key,
        max_chars=MAX_OPERATION_KEY_CHARS,
        allow_empty=False,
        redaction_env=env,
    )
    if not isinstance(kind, str) or kind not in _RUN_KINDS:
        raise ValueError(f"invalid operation kind: {kind!r}")
    if not isinstance(source_agent, str) or source_agent not in _SOURCE_AGENTS:
        raise ValueError(f"invalid operation source_agent: {source_agent!r}")
    session = _validate_identity_text(
        "session_id",
        session_id,
        max_chars=MAX_OPERATION_IDENTITY_CHARS,
        allow_empty=source_agent == "system",
        redaction_env=env,
    )
    project_name = _validate_identity_text(
        "project",
        project,
        max_chars=MAX_OPERATION_IDENTITY_CHARS,
        allow_empty=False,
        redaction_env=env,
    )
    return (
        key,
        cast(RunKind, kind),
        cast(SourceAgent, source_agent),
        session,
        project_name,
    )


def validate_operation_event(state: RunState, phase: str) -> None:
    """Reject informative events incompatible with an operation's current state."""
    if state in _TERMINAL_STATES:
        raise ValueError(f"terminal operation state {state!r} cannot accept events")
    if phase not in _OPERATION_PHASES[state]:
        raise ValueError(f"status phase {phase!r} is incompatible with state {state!r}")


def validate_operation_transition(
    current_state: RunState,
    target_state: RunState,
    phase: str,
) -> None:
    """Validate an explicit operation state change and its target phase."""
    if current_state in _TERMINAL_STATES:
        raise ValueError(
            f"terminal operation state {current_state!r} cannot transition"
        )
    if target_state not in _OPERATION_TRANSITIONS[current_state]:
        raise ValueError(
            f"invalid operation transition {current_state!r} -> {target_state!r}"
        )
    if phase not in _OPERATION_PHASES[target_state]:
        raise ValueError(
            f"status phase {phase!r} is incompatible with state {target_state!r}"
        )


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
    timeline_available: bool = True
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
    provider: ProviderName | None
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
        if self.provider is not None and self.provider not in _PROVIDER_NAMES:
            raise ValueError(f"invalid status event provider: {self.provider!r}")
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


@dataclass(frozen=True)
class HealthAlert:
    """One already-filtered operational health warning injected by a caller."""

    created_at: datetime
    level: EventLevel
    message: str
    component: str = "hook"
    redaction_env: InitVar[Mapping[str, str] | None] = None

    def __post_init__(self, redaction_env: Mapping[str, str] | None) -> None:
        if self.level not in _EVENT_LEVELS:
            raise ValueError(f"invalid health alert level: {self.level!r}")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("health alert timestamps must be timezone-aware")
        env = redaction_env or os.environ
        try:
            component = _validate_identity_text(
                "health alert component",
                self.component,
                max_chars=MAX_OPERATION_IDENTITY_CHARS,
                allow_empty=False,
                redaction_env=env,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("health alert component is unsafe") from error
        message = normalize_event_message(self.message, env)
        if message is None:
            raise ValueError("health alert message must not be empty")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "message", message)


@dataclass(frozen=True)
class ObserverState:
    """Display-only state that never changes queue execution."""

    version: int
    acknowledged_run_ids: frozenset[int]

    @classmethod
    def empty(cls) -> ObserverState:
        return cls(version=1, acknowledged_run_ids=frozenset())

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("observer state version must be 1")
        if any(
            isinstance(run_id, bool) or not isinstance(run_id, int) or run_id == 0
            for run_id in self.acknowledged_run_ids
        ):
            raise ValueError("acknowledged run identifiers must be nonzero integers")


@dataclass(frozen=True)
class CompileStatus:
    """Current end-of-day compile state for snapshot renderers."""

    state: str
    summary: str
    run: StatusRun | None = None
    ready: bool = False


@dataclass(frozen=True)
class CompileReadinessProbes:
    """Injected, side-effect-free inputs for compile readiness projection."""

    local_now: Callable[[], datetime]
    session_count: Callable[[], int]
    daily_state: Callable[[], str]
    reservation_state: Callable[[], str | None]


@dataclass(frozen=True)
class ProviderAttempt:
    """Privacy-safe provider attempt metadata for one run."""

    id: int
    provider: str
    outcome: str
    reason: str | None
    started_at: datetime
    ended_at: datetime
    elapsed_ms: int


@dataclass(frozen=True)
class RunDetails:
    """One run and its immutable, content-free operational timeline."""

    run: StatusRun
    events: tuple[StatusEvent, ...]
    provider_attempts: tuple[ProviderAttempt, ...]
    timeline_available: bool


@dataclass(frozen=True)
class StatusSnapshot:
    """Immutable dashboard projection grouped for operational triage."""

    active: tuple[StatusRun, ...]
    attention: tuple[StatusRun, ...]
    recent: tuple[StatusRun, ...]
    compile: CompileStatus
    health_alerts: tuple[HealthAlert, ...]
    has_more: bool = False


class StatusReadError(RuntimeError):
    """Base class for typed read-only dashboard diagnostics."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = Path(os.path.abspath(path.expanduser()))
        super().__init__(message)


class StatusDatabaseUnavailable(StatusReadError):
    """Raised when the configured queue cannot be opened read-only."""


class StatusDatabaseUnsafe(StatusReadError):
    """Raised when the queue path cannot be identity-pinned as private."""


class StatusDataInvalid(StatusReadError):
    """Raised when persisted queue/status data cannot be safely projected."""


MAX_OBSERVER_STATE_BYTES = 64 * 1024
MAX_ACKNOWLEDGED_RUN_IDS = 2_048
_OBSERVER_STATE_RELATIVE_PATH = Path("scripts/status-state/status-view.json")


def observer_state_path(memory_home: Path) -> Path:
    """Return the private display-state path for one configured memory root."""
    return Path(os.path.abspath(memory_home.expanduser())) / _OBSERVER_STATE_RELATIVE_PATH


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
    state: RunState = "queued",
    phase: str = "queued",
    summary: str | None = None,
    error: str | None = None,
    updated_at: datetime | None = None,
    completed_at: datetime | None = None,
    redaction_env: Mapping[str, str] | None = None,
) -> int:
    """Create a validated job run using the caller's existing transaction."""
    candidate = StatusRun(
        id=0,
        job_id=job_id,
        operation_key=None,
        kind=kind,
        source_agent=source_agent,
        session_id=session_id,
        project=project,
        state=state,
        phase=phase,
        summary=summary,
        error=error,
        started_at=now,
        updated_at=updated_at or now,
        completed_at=completed_at,
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
            (
                _stored_time(candidate.completed_at)
                if candidate.completed_at is not None
                else None
            ),
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("created status run has no identifier")
    return cursor.lastrowid


def create_operation_run_unlocked(
    connection: sqlite3.Connection,
    *,
    operation_key: str,
    kind: RunKind,
    source_agent: SourceAgent,
    session_id: str,
    project: str,
    phase: str,
    now: datetime,
    redaction_env: Mapping[str, str] | None = None,
) -> int:
    """Create a queued non-job run using the caller's transaction."""
    operation_key, kind, source_agent, session_id, project = validate_operation_identity(
        operation_key,
        kind,
        source_agent,
        session_id,
        project,
        redaction_env=redaction_env,
    )
    validate_operation_event("queued", phase)
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
    provider: ProviderName | None = None,
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
    provider: ProviderName | None = None,
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


_ACTIVE_RUN_STATES = frozenset({"queued", "running", "retrying"})
_ATTENTION_RUN_STATES = frozenset({"failed", "dead"})
_RECENT_WINDOW = timedelta(days=7)


def _open_read_only_database(queue_path: Path) -> sqlite3.Connection:
    resolved = Path(os.path.abspath(queue_path.expanduser()))
    try:
        expected_identity = inspect_secure_read_file(resolved)
    except FileNotFoundError as error:
        raise StatusDatabaseUnavailable(
            resolved,
            f"status database is unavailable: {resolved}",
        ) from error
    except (OSError, ValueError) as error:
        raise StatusDatabaseUnsafe(
            resolved,
            f"status database path is unsafe: {resolved}",
        ) from error
    resolved = expected_identity.path
    try:
        connection = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro",
            uri=True,
            timeout=0.1,
        )
    except sqlite3.Error as error:
        raise StatusDatabaseUnavailable(
            resolved,
            f"status database is unavailable: {resolved}",
        ) from error
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 100")
        try:
            observed_identity = inspect_secure_read_file(resolved)
        except (OSError, ValueError) as error:
            raise StatusDatabaseUnsafe(
                resolved,
                f"status database path became unsafe during open: {resolved}",
            ) from error
        if (
            observed_identity.device,
            observed_identity.inode,
        ) != (expected_identity.device, expected_identity.inode):
            raise StatusDatabaseUnsafe(
                resolved,
                f"status database identity changed during open: {resolved}",
            )
        return connection
    except BaseException:
        connection.close()
        raise


def _database_tables(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    )


def _synthetic_run_from_job(row: sqlite3.Row) -> StatusRun:
    state_by_job_status: Mapping[str, tuple[RunState, str]] = {
        "pending": ("queued", "queued"),
        "leased": ("running", "worker_claimed"),
        "failed": ("retrying", "retry_wait"),
        "succeeded": ("succeeded", "succeeded"),
        "dead": ("dead", "dead"),
    }
    job_status = str(row["status"])
    try:
        state, phase = state_by_job_status[job_status]
    except KeyError as error:
        raise ValueError(f"unsupported legacy job status: {job_status!r}") from error
    started_at = _loaded_time(row["created_at"])
    updated_at = _loaded_time(row["updated_at"])
    if started_at is None or updated_at is None:
        raise ValueError("legacy job is missing a required timestamp")
    summary_by_state = {
        "queued": "Queued",
        "running": "Worker claimed",
        "retrying": "Waiting to retry",
        "succeeded": "Completed",
        "dead": "Attempts exhausted",
    }
    job_id = int(row["id"])
    return StatusRun(
        id=-job_id,
        job_id=job_id,
        operation_key=None,
        kind=_safe_projection_identity(
            "kind", row["kind"], max_chars=64, allowed=_RUN_KINDS
        ),
        source_agent=_safe_projection_identity(
            "source_agent",
            row["source_agent"],
            max_chars=32,
            allowed=_SOURCE_AGENTS,
        ),
        session_id=_safe_projection_identity(
            "session_id",
            row["session_id"],
            max_chars=MAX_OPERATION_IDENTITY_CHARS,
        ),
        project=_safe_projection_identity(
            "project",
            row["project"],
            max_chars=MAX_OPERATION_IDENTITY_CHARS,
        ),
        state=state,
        phase=phase,
        summary=summary_by_state[state],
        error=row["last_error"],
        started_at=started_at,
        updated_at=updated_at,
        completed_at=_loaded_time(row["completed_at"]),
        timeline_available=False,
        redaction_env=os.environ,
    )


def _safe_projection_identity(
    field: str,
    value: object,
    *,
    max_chars: int,
    allowed: frozenset[str] | None = None,
) -> str:
    if not isinstance(value, str) or (allowed is not None and value not in allowed):
        return "unknown"
    try:
        return _validate_identity_text(
            field,
            value,
            max_chars=max_chars,
            allow_empty=False,
            redaction_env=os.environ,
        )
    except (TypeError, ValueError):
        return "unknown"


def _safe_projection_run(run: StatusRun) -> StatusRun:
    return replace(
        run,
        operation_key=_safe_projection_operation_key(run.operation_key),
        kind=_safe_projection_identity(
            "kind", run.kind, max_chars=64, allowed=_RUN_KINDS
        ),
        source_agent=_safe_projection_identity(
            "source_agent",
            run.source_agent,
            max_chars=32,
            allowed=_SOURCE_AGENTS,
        ),
        session_id=_safe_projection_identity(
            "session_id",
            run.session_id,
            max_chars=MAX_OPERATION_IDENTITY_CHARS,
        ),
        project=_safe_projection_identity(
            "project",
            run.project,
            max_chars=MAX_OPERATION_IDENTITY_CHARS,
        ),
        redaction_env=os.environ,
    )


def _safe_projection_operation_key(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    try:
        return _validate_identity_text(
            "operation_key",
            value,
            max_chars=MAX_OPERATION_KEY_CHARS,
            allow_empty=False,
            redaction_env=os.environ,
        )
    except (TypeError, ValueError):
        return None


def _read_runs(
    connection: sqlite3.Connection,
    tables: frozenset[str],
    *,
    now: datetime,
    all_history: bool,
    max_runs: int | None,
    query: str,
    acknowledged_run_ids: frozenset[int],
) -> tuple[StatusRun, ...]:
    runs: list[StatusRun] = []
    cutoff = _stored_time(now.astimezone(UTC) - _RECENT_WINDOW)
    needle = " ".join(query.split()).casefold()
    def safe_match(*values: object) -> int:
        return int(any(needle in (normalize_summary(value, os.environ) or "").casefold() for value in values))
    connection.create_function("safe_status_match", -1, safe_match, deterministic=True)
    def safe_legacy_match(*values: object) -> int:
        kind, source, session, project, status = values
        mapping = {
            "pending": ("queued", "queued", "Queued"),
            "leased": ("running", "worker_claimed", "Worker claimed"),
            "failed": ("retrying", "retry_wait", "Waiting to retry"),
            "succeeded": ("succeeded", "succeeded", "Completed"),
            "dead": ("dead", "dead", "Attempts exhausted"),
        }.get(str(status), ("unknown", "unknown", ""))
        return safe_match(kind, source, session, project, *mapping)
    connection.create_function("safe_legacy_match", 5, safe_legacy_match, deterministic=True)
    if "status_runs" in tables:
        status_sql = "SELECT * FROM status_runs WHERE kind != 'compile'"
        status_parameters: tuple[object, ...] = ()
        if not all_history:
            status_sql += (
                " AND (state IN ('queued', 'running', 'retrying', 'failed', 'dead')"
                " OR (state = 'succeeded'"
                " AND julianday(updated_at) >= julianday(?)))"
            )
            status_parameters = (cutoff,)
        if needle:
            status_sql += " AND " + (
                "safe_status_match(kind,source_agent,session_id,project,state,phase,summary)=1"
            )
        status_sql += (
            " ORDER BY CASE WHEN state IN ('failed','dead') AND id NOT IN ("
            + (",".join("?" for _ in acknowledged_run_ids) or "NULL")
            + ") THEN 0 "
            "WHEN state IN ('queued','running','retrying') THEN 1 ELSE 2 END, "
            "julianday(updated_at) DESC, id DESC"
        )
        status_parameters = (*status_parameters, *sorted(acknowledged_run_ids))
        if max_runs is not None:
            status_sql += " LIMIT ?"
            status_parameters = (*status_parameters, max_runs + 1)
        runs.extend(
            _safe_projection_run(status_run_from_row(row, redaction_env=os.environ))
            for row in connection.execute(status_sql, status_parameters)
        )
        legacy_sql = """
            SELECT
                jobs.id, jobs.kind, jobs.source_agent, jobs.session_id,
                jobs.project, jobs.status, jobs.last_error, jobs.created_at,
                jobs.updated_at, jobs.completed_at
            FROM jobs
            LEFT JOIN status_runs ON status_runs.job_id = jobs.id
            WHERE status_runs.id IS NULL AND jobs.kind != 'compile'
            """
        legacy_parameters: tuple[object, ...] = ()
        if not all_history:
            legacy_sql += (
                " AND (jobs.status IN ('pending', 'leased', 'failed', 'dead')"
                " OR (jobs.status = 'succeeded'"
                " AND julianday(jobs.updated_at) >= julianday(?)))"
            )
            legacy_parameters = (cutoff,)
        if needle:
            legacy_sql += " AND safe_legacy_match(jobs.kind,jobs.source_agent,jobs.session_id,jobs.project,jobs.status)=1"
        legacy_sql += (
            " ORDER BY CASE WHEN jobs.status IN ('failed','dead') THEN 0 "
            "WHEN jobs.status IN ('pending','leased') THEN 1 ELSE 2 END, "
            "julianday(jobs.updated_at) DESC, jobs.id DESC"
        )
        if max_runs is not None:
            legacy_sql += " LIMIT ?"
            legacy_parameters = (*legacy_parameters, max_runs + 1)
        legacy_rows = connection.execute(legacy_sql, legacy_parameters)
    else:
        legacy_sql = """
            SELECT id, kind, source_agent, session_id, project, status,
                last_error, created_at, updated_at, completed_at
            FROM jobs WHERE kind != 'compile'
            """
        legacy_parameters = ()
        if not all_history:
            legacy_sql += (
                " AND (status IN ('pending', 'leased', 'failed', 'dead')"
                " OR (status = 'succeeded'"
                " AND julianday(updated_at) >= julianday(?)))"
            )
            legacy_parameters = (cutoff,)
        if needle:
            legacy_sql += " AND safe_legacy_match(kind,source_agent,session_id,project,status)=1"
        legacy_sql += (
            " ORDER BY CASE WHEN status IN ('failed','dead') THEN 0 "
            "WHEN status IN ('pending','leased') THEN 1 ELSE 2 END, "
            "julianday(updated_at) DESC, id DESC"
        )
        if max_runs is not None:
            legacy_sql += " LIMIT ?"
            legacy_parameters = (*legacy_parameters, max_runs + 1)
        legacy_rows = connection.execute(legacy_sql, legacy_parameters)
    runs.extend(_synthetic_run_from_job(row) for row in legacy_rows)
    return tuple(runs)


def _read_latest_compile_run(
    connection: sqlite3.Connection,
    tables: frozenset[str],
) -> StatusRun | None:
    if "status_runs" not in tables:
        return None
    row = connection.execute(
        """
        SELECT * FROM status_runs
        WHERE kind = 'compile'
        ORDER BY julianday(updated_at) DESC, id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return _safe_projection_run(
        status_run_from_row(row, redaction_env=os.environ)
    )


def _matches_query(run: StatusRun, query: str) -> bool:
    needle = " ".join(query.split()).casefold()
    if not needle:
        return True
    approved_fields = (
        run.kind,
        run.source_agent,
        run.session_id,
        run.project,
        run.state,
        run.phase,
        run.summary or "",
    )
    return any(needle in field.casefold() for field in approved_fields)


def _run_sort_key(run: StatusRun) -> tuple[datetime, int]:
    return run.updated_at, run.id


def project_compile_status(
    *,
    compile_run: StatusRun | None,
    queue_active_count: int,
    probes: CompileReadinessProbes,
) -> CompileStatus:
    """Purely project compile readiness from injected operational probes."""
    if compile_run is not None and compile_run.state in _ACTIVE_RUN_STATES:
        summary = (
            compile_run.summary
            or compile_run.error
            or compile_run.phase.replace("_", " ").capitalize()
        )
        return CompileStatus(
            state=compile_run.state,
            summary=summary,
            run=compile_run,
            ready=False,
        )

    def with_last_result(status: CompileStatus) -> CompileStatus:
        return replace(status, run=compile_run) if compile_run is not None else status

    reservation = probes.reservation_state()
    if reservation is not None:
        reservation_state = {
            "failed": "failed",
            "retry_wait": "retrying",
            "queue_wait": "retrying",
            "read_wait": "retrying",
        }.get(reservation, "reserved")
        return with_last_result(
            CompileStatus(
                state=reservation_state,
                summary=(
                    "Automatic compile is waiting to retry"
                    if reservation_state == "retrying"
                    else "Automatic compile is reserved"
                ),
            ),
        )
    local_now = probes.local_now()
    if local_now.tzinfo is None or local_now.utcoffset() is None:
        raise ValueError("compile readiness time must be timezone-aware")
    if local_now.hour < 16:
        return with_last_result(
            CompileStatus(
                state="before_window",
                summary="Next automatic compile window begins at 16:00",
            )
        )
    if queue_active_count < 0:
        raise ValueError("queue active count must be nonnegative")
    if queue_active_count:
        return with_last_result(
            CompileStatus(
                state="waiting_queue",
                summary=f"Waiting for {queue_active_count} flush job(s)",
            )
        )
    session_count = probes.session_count()
    if session_count < 0:
        return with_last_result(
            CompileStatus(
                state="unavailable",
                summary="Interactive session count is unavailable",
            )
        )
    if session_count:
        return with_last_result(
            CompileStatus(
                state="waiting_sessions",
                summary=f"Waiting for {session_count} interactive session(s) to close",
            )
        )
    daily_state = probes.daily_state()
    if daily_state == "covered":
        return with_last_result(
            CompileStatus(
                state="complete",
                summary="Today's captured content is compiled",
            )
        )
    if daily_state == "uncompiled":
        return with_last_result(
            CompileStatus(
                state="ready",
                summary="Automatic compile is ready",
                ready=True,
            )
        )
    return with_last_result(
        CompileStatus(
            state="unavailable",
            summary="Today's compile state is unavailable",
        )
    )


def _read_compile_database_state(
    connection: sqlite3.Connection,
    tables: frozenset[str],
    now: datetime,
) -> tuple[int, str | None]:
    queue_active_count = connection.execute(
        "SELECT count(*) FROM jobs WHERE status IN ('pending', 'leased', 'failed')"
    ).fetchone()[0]
    reservation_state: str | None = None
    if "queue_metadata" in tables:
        row = connection.execute(
            "SELECT value FROM queue_metadata WHERE key = 'auto_compile_reservation'"
        ).fetchone()
        if row is not None:
            try:
                reservation = json.loads(row["value"])
            except (TypeError, json.JSONDecodeError):
                reservation = None
            if isinstance(reservation, dict):
                expiry_values = []
                for key in (
                    "expires_at",
                    "watcher_expires_at",
                    "contender_expires_at",
                    "next_retry_at",
                ):
                    value = reservation.get(key)
                    if isinstance(value, str):
                        try:
                            expiry = _loaded_time(value)
                        except ValueError:
                            continue
                        if expiry is not None:
                            expiry_values.append(expiry)
                raw_state = reservation.get("status", "reserved")
                if raw_state == "failed" or any(
                    expiry > now.astimezone(UTC) for expiry in expiry_values
                ):
                    reservation_state = (
                        raw_state if isinstance(raw_state, str) else "reserved"
                    )
    return int(queue_active_count), reservation_state


def _default_session_count() -> int:
    try:
        if __package__:
            from .flush import count_interactive_agent_sessions
        else:
            from flush import count_interactive_agent_sessions

        return count_interactive_agent_sessions()
    except (ImportError, OSError, RuntimeError):
        return -1


def _default_daily_state(
    queue_path: Path,
    now: datetime,
    memory_home: Path | None = None,
) -> str:
    try:
        if __package__:
            from .flush import _read_daily_compile_state
        else:
            from flush import _read_daily_compile_state

        root = (
            Path(os.path.abspath(memory_home.expanduser()))
            if memory_home is not None
            else (
                queue_path.parent.parent
                if queue_path.parent.name == "scripts"
                else None
            )
        )
        if root is None:
            return "unreadable"
        local_now = now.astimezone()
        daily_path = (
            root / "daily" / f"{local_now.strftime('%Y-%m-%d')}.md"
        )
        return _read_daily_compile_state(daily_path).status
    except (ImportError, OSError, RuntimeError, ValueError):
        return "unreadable"


def read_snapshot(
    queue_path: Path,
    *,
    now: datetime,
    observer_state: ObserverState,
    query: str = "",
    health_alerts: tuple[HealthAlert, ...] = (),
    memory_home: Path | None = None,
    max_runs: int | None = None,
) -> StatusSnapshot:
    """Read and group status without creating, migrating, or mutating the queue."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("snapshot time must be timezone-aware")
    if max_runs is not None and (
        isinstance(max_runs, bool) or not isinstance(max_runs, int) or not 1 <= max_runs <= 10_000
    ):
        raise ValueError("max_runs must be between 1 and 10000")
    resolved = Path(os.path.abspath(queue_path.expanduser()))
    connection = _open_read_only_database(resolved)
    try:
        tables = _database_tables(connection)
        if "jobs" not in tables:
            raise StatusDataInvalid(resolved, "status database has no jobs table")
        connection.execute("BEGIN")
        runs = _read_runs(
            connection,
            tables,
            now=now,
            all_history=bool(query.strip()),
            max_runs=max_runs,
            query=query,
            acknowledged_run_ids=observer_state.acknowledged_run_ids,
        )
        latest_compile_run = _read_latest_compile_run(connection, tables)
        queue_active_count, reservation_state = _read_compile_database_state(
            connection, tables, now
        )
    except StatusReadError:
        raise
    except (json.JSONDecodeError, sqlite3.Error, TypeError, ValueError) as error:
        raise StatusDataInvalid(
            resolved,
            f"status database contains invalid operational data: {resolved}",
        ) from error
    finally:
        connection.close()

    matching = tuple(sorted((run for run in runs if _matches_query(run, query)), key=_run_sort_key, reverse=True))
    active = tuple(
        sorted(
            (run for run in matching if run.state in _ACTIVE_RUN_STATES),
            key=_run_sort_key,
            reverse=True,
        )
    )
    attention = tuple(
        sorted(
            (
                run
                for run in matching
                if run.state in _ATTENTION_RUN_STATES
                and run.id not in observer_state.acknowledged_run_ids
            ),
            key=_run_sort_key,
            reverse=True,
        )
    )
    cutoff = now.astimezone(UTC) - _RECENT_WINDOW
    search_all_history = bool(query.strip())
    recent = tuple(
        sorted(
            (
                run
                for run in matching
                if (
                    run.state == "succeeded"
                    or (
                        run.state in _ATTENTION_RUN_STATES
                        and run.id in observer_state.acknowledged_run_ids
                    )
                )
                and (search_all_history or run.updated_at >= cutoff)
            ),
            key=_run_sort_key,
            reverse=True,
        )
    )
    compile_runs_by_id = {run.id: run for run in runs if run.kind == "compile"}
    if latest_compile_run is not None:
        compile_runs_by_id[latest_compile_run.id] = latest_compile_run
    compile_runs = tuple(compile_runs_by_id.values())
    compile_run = max(compile_runs, key=_run_sort_key) if compile_runs else None
    compile_status = project_compile_status(
        compile_run=compile_run,
        queue_active_count=queue_active_count,
        probes=CompileReadinessProbes(
            local_now=lambda: now.astimezone(),
            session_count=_default_session_count,
            daily_state=(
                (lambda: _default_daily_state(resolved, now, memory_home))
                if memory_home is not None
                else (lambda: _default_daily_state(resolved, now))
            ),
            reservation_state=lambda: reservation_state,
        ),
    )
    has_more = False
    if max_runs is not None:
        has_more = len(active) + len(attention) + len(recent) > max_runs
        sources = (attention, active, recent)
        targets: list[list[StatusRun]] = [[], [], []]
        nonempty = [index for index, source in enumerate(sources) if source]
        initial = nonempty if max_runs >= len(nonempty) else nonempty[:max_runs]
        for index in initial:
            targets[index].append(sources[index][0])
        remaining = max_runs - len(initial)
        for index in (0, 1, 2):
            if remaining <= 0:
                break
            extra = sources[index][len(targets[index]): len(targets[index]) + remaining]
            targets[index].extend(extra)
            remaining -= len(extra)
        attention, active, recent = map(tuple, targets)
    return StatusSnapshot(
        active=active,
        attention=attention,
        recent=recent,
        compile=compile_status,
        health_alerts=tuple(health_alerts),
        has_more=has_more,
    )


def _provider_attempt_from_row(row: sqlite3.Row) -> ProviderAttempt:
    started_at = _loaded_time(row["started_at"])
    ended_at = _loaded_time(row["ended_at"])
    if started_at is None or ended_at is None:
        raise ValueError("provider attempt is missing a required timestamp")
    provider = str(row["provider"])
    if provider not in _PROVIDER_NAMES:
        provider = "unknown"
    raw_outcome = row["outcome"]
    outcome = (
        raw_outcome
        if raw_outcome
        in {"success", "auth_failed", "capacity", "timeout", "invalid_output", "error"}
        else "unknown"
    )
    elapsed_ms = row["elapsed_ms"]
    if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int) or elapsed_ms < 0:
        raise ValueError("provider attempt elapsed_ms must be a nonnegative integer")
    return ProviderAttempt(
        id=row["id"],
        provider=provider,
        outcome=outcome,
        reason=normalize_status_reason(row["reason"], os.environ),
        started_at=started_at,
        ended_at=ended_at,
        elapsed_ms=elapsed_ms,
    )


def read_run_details(queue_path: Path, run_id: int) -> RunDetails:
    """Read one run timeline without exposing queue payload or provider output."""
    resolved = Path(os.path.abspath(queue_path.expanduser()))
    connection = _open_read_only_database(resolved)
    try:
        tables = _database_tables(connection)
        if "jobs" not in tables:
            raise StatusDataInvalid(resolved, "status database has no jobs table")
        connection.execute("BEGIN")
        if run_id > 0 and "status_runs" in tables:
            row = connection.execute(
                "SELECT * FROM status_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = _safe_projection_run(
                status_run_from_row(row, redaction_env=os.environ)
            )
            events = tuple(
                status_event_from_row(event, redaction_env=os.environ)
                for event in connection.execute(
                    "SELECT * FROM status_events WHERE run_id = ? ORDER BY id",
                    (run_id,),
                )
            )
        else:
            job_id = -run_id if run_id < 0 else run_id
            row = connection.execute(
                """
                SELECT id, kind, source_agent, session_id, project, status,
                    last_error, created_at, updated_at, completed_at
                FROM jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = _synthetic_run_from_job(row)
            events = ()
        attempts = (
            tuple(
                _provider_attempt_from_row(attempt)
                for attempt in connection.execute(
                    """
                    SELECT id, provider, outcome, reason, started_at, ended_at,
                        elapsed_ms
                    FROM provider_attempts WHERE job_id = ? ORDER BY id
                    """,
                    (run.job_id,),
                )
            )
            if run.job_id is not None and "provider_attempts" in tables
            else ()
        )
        return RunDetails(
            run=run,
            events=events,
            provider_attempts=attempts,
            timeline_available=run.timeline_available,
        )
    except (KeyError, StatusReadError):
        raise
    except (json.JSONDecodeError, sqlite3.Error, TypeError, ValueError) as error:
        raise StatusDataInvalid(
            resolved,
            f"status database contains invalid operational data: {resolved}",
        ) from error
    finally:
        connection.close()


def _prune_acknowledged_run_ids(
    identifiers: frozenset[int],
    *,
    current_run_id: int | None = None,
) -> frozenset[int]:
    if len(identifiers) <= MAX_ACKNOWLEDGED_RUN_IDS:
        return identifiers
    retained = sorted(
        identifiers,
        key=lambda run_id: (
            run_id == current_run_id,
            abs(run_id),
            run_id,
        ),
        reverse=True,
    )[:MAX_ACKNOWLEDGED_RUN_IDS]
    return frozenset(retained)


def _load_observer_state_unlocked(path: Path) -> ObserverState:
    try:
        data = read_private_bounded_file(path, max_bytes=MAX_OBSERVER_STATE_BYTES)
        if data is None:
            return ObserverState.empty()
        decoded = json.loads(data.decode("utf-8"))
        if not isinstance(decoded, dict) or set(decoded) != {
            "version",
            "acknowledged_run_ids",
        }:
            return ObserverState.empty()
        identifiers = decoded["acknowledged_run_ids"]
        if (
            decoded["version"] != 1
            or isinstance(decoded["version"], bool)
            or not isinstance(identifiers, list)
            or any(
                isinstance(run_id, bool)
                or not isinstance(run_id, int)
                or run_id == 0
                for run_id in identifiers
            )
        ):
            return ObserverState.empty()
        return ObserverState(
            version=1,
            acknowledged_run_ids=_prune_acknowledged_run_ids(
                frozenset(identifiers)
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return ObserverState.empty()


def load_observer_state(path: Path) -> ObserverState:
    """Load exact version-1 display state; unsafe or malformed files reset safely."""
    return _load_observer_state_unlocked(path)


def acknowledge_run(path: Path, run_id: int) -> ObserverState:
    """Persist one display-only acknowledgment with a private atomic replace."""
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id == 0:
        raise ValueError("acknowledged run identifier must be a nonzero integer")
    with secure_private_state_lock(path):
        current = _load_observer_state_unlocked(path)
        identifiers = _prune_acknowledged_run_ids(
            current.acknowledged_run_ids | {run_id},
            current_run_id=run_id,
        )
        updated = ObserverState(version=1, acknowledged_run_ids=identifiers)
        payload = json.dumps(
            {
                "version": 1,
                "acknowledged_run_ids": sorted(updated.acknowledged_run_ids),
            },
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
        atomic_write_private_file(path, payload, max_bytes=MAX_OBSERVER_STATE_BYTES)
        return updated
