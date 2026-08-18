"""Durable SQLite queue for memory compiler jobs and provider attempts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import sysconfig
from typing import Callable, Literal, Mapping, TypeAlias

try:
    from .privacy import normalize_persistence_reason
    from .providers import ProviderResult
    from .status_store import (
        ALLOWED_PHASES,
        EventLevel,
        JsonScalar,
        ProviderName,
        RunKind,
        RunState,
        SourceAgent,
        StatusEvent,
        StatusRun,
        append_event_unlocked,
        create_job_run_unlocked,
        create_operation_run_unlocked,
        status_event_from_row,
        status_run_from_row,
        transition_run_unlocked,
        validate_operation_event,
        validate_operation_identity,
        validate_operation_transition,
    )
    from .transcripts import NormalizedSession, render_turns
    from .usage import (
        UnsafeUsagePathError,
        UsageRecord,
        append_usage_record,
        logged_provider_attempt_ids,
        recover_usage_log,
    )
except ImportError:  # Direct execution with scripts/ on sys.path.
    from privacy import normalize_persistence_reason
    from providers import ProviderResult
    from status_store import (
        ALLOWED_PHASES,
        EventLevel,
        JsonScalar,
        ProviderName,
        RunKind,
        RunState,
        SourceAgent,
        StatusEvent,
        StatusRun,
        append_event_unlocked,
        create_job_run_unlocked,
        create_operation_run_unlocked,
        status_event_from_row,
        status_run_from_row,
        transition_run_unlocked,
        validate_operation_event,
        validate_operation_identity,
        validate_operation_transition,
    )
    from transcripts import NormalizedSession, render_turns
    from usage import (
        UnsafeUsagePathError,
        UsageRecord,
        append_usage_record,
        logged_provider_attempt_ids,
        recover_usage_log,
    )


# The repository adds scripts/ to sys.path, so this required filename can shadow
# Python's standard ``queue`` module. Preserve its public classes for libraries
# (notably AnyIO) that import them after this module has loaded.
_stdlib_queue_path = Path(sysconfig.get_path("stdlib")) / "queue.py"
_stdlib_queue_spec = importlib.util.spec_from_file_location(
    "_ai_memory_stdlib_queue", _stdlib_queue_path
)
if _stdlib_queue_spec is None or _stdlib_queue_spec.loader is None:
    raise ImportError(f"could not load standard queue module from {_stdlib_queue_path}")
_stdlib_queue = importlib.util.module_from_spec(_stdlib_queue_spec)
_stdlib_queue_spec.loader.exec_module(_stdlib_queue)
Empty = _stdlib_queue.Empty
Full = _stdlib_queue.Full
Queue = _stdlib_queue.Queue
PriorityQueue = _stdlib_queue.PriorityQueue
LifoQueue = _stdlib_queue.LifoQueue
SimpleQueue = _stdlib_queue.SimpleQueue


JobStatus = Literal["pending", "leased", "succeeded", "failed", "dead"]
AutoCompileReadStatus = Literal["unreadable", "covered", "uncompiled"]
AutoCompileContentRead: TypeAlias = (
    tuple[AutoCompileReadStatus, str | None]
    | tuple[AutoCompileReadStatus, str | None, tuple[str, ...]]
)
SCHEMA_VERSION = 3
DEFAULT_MAX_ATTEMPTS = 5
AUTO_COMPILE_RESERVATION_KEY = "auto_compile_reservation"

_SYNTHESIZED_JOB_STATUS: dict[
    JobStatus,
    tuple[RunState, str, EventLevel, bool],
] = {
    "pending": ("queued", "queued", "info", False),
    "leased": ("running", "worker_claimed", "info", False),
    "failed": ("retrying", "retry_wait", "warning", True),
    "succeeded": ("succeeded", "succeeded", "info", False),
    "dead": ("dead", "dead", "error", True),
}

_JOB_EVENT_PHASES = frozenset(
    {
        "codex_started",
        "codex_succeeded",
        "codex_failed",
        "claude_started",
        "claude_succeeded",
        "claude_failed",
        "daily_log_write_started",
    }
)

_STATUS_SCHEMA_STATEMENTS = (
    """
CREATE TABLE status_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    operation_key TEXT UNIQUE,
    kind TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    session_id TEXT NOT NULL,
    project TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('queued', 'running', 'retrying', 'succeeded', 'failed', 'dead')
    ),
    phase TEXT NOT NULL,
    summary TEXT,
    error TEXT,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK ((job_id IS NULL) <> (operation_key IS NULL))
)""",
    """
CREATE TABLE status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES status_runs(id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('info', 'warning', 'error')),
    provider TEXT,
    attempt INTEGER,
    message TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
)""",
    "CREATE INDEX status_runs_state_updated_idx ON status_runs(state, updated_at DESC)",
    "CREATE INDEX status_events_run_id_id_idx ON status_events(run_id, id)",
)

_QUEUE_V2_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL CHECK (
            kind IN ('capture', 'compile', 'query_file', 'connections', 'semantic_lint')
        ),
        source_agent TEXT NOT NULL CHECK (
            source_agent IN ('claude', 'codex', 'system')
        ),
        session_id TEXT NOT NULL,
        project TEXT NOT NULL,
        cwd TEXT NOT NULL,
        trigger TEXT NOT NULL,
        source_path TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending' CHECK (
            status IN ('pending', 'leased', 'succeeded', 'failed', 'dead')
        ),
        attempt_count INTEGER NOT NULL DEFAULT 0,
        available_at TEXT NOT NULL,
        lease_owner TEXT,
        lease_expires_at TEXT,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        UNIQUE (kind, source_agent, session_id, source_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        task TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT NOT NULL,
        outcome TEXT NOT NULL,
        reason TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        elapsed_ms INTEGER NOT NULL,
        legacy_cost_usd REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS jobs_status_available_idx ON jobs(status, available_at)",
    "CREATE INDEX IF NOT EXISTS jobs_lease_expiry_idx ON jobs(lease_expires_at)",
    "CREATE TABLE IF NOT EXISTS queue_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """
    INSERT OR IGNORE INTO queue_metadata(key, value)
    VALUES ('queue_id', lower(hex(randomblob(16))))
    """,
)

_VERSION_1_TO_2_STATEMENTS = (
    "ALTER TABLE provider_attempts ADD COLUMN legacy_cost_usd REAL",
    "CREATE TABLE queue_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """
    INSERT INTO queue_metadata(key, value)
    VALUES ('queue_id', lower(hex(randomblob(16))))
    """,
)


class LeaseOwnershipError(RuntimeError):
    """Raised when a worker mutates a lease it does not own."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime(value: datetime | str | int | float) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(value, timezone.utc)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _stored_time(value: datetime | str | int | float) -> str:
    return _datetime(value).isoformat(timespec="microseconds")


def _loaded_time(value: str | None) -> datetime | None:
    return _datetime(value) if value is not None else None


@dataclass(frozen=True)
class Job:
    id: int
    kind: str
    source_agent: str
    session_id: str
    project: str
    cwd: str
    trigger: str
    source_path: str
    source_hash: str
    payload_json: str
    status: JobStatus
    attempt_count: int
    available_at: datetime
    lease_owner: str | None
    lease_expires_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @property
    def payload(self) -> dict:
        value = json.loads(self.payload_json)
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ProviderAttempt:
    id: int
    job_id: int
    provider: str
    model: str
    task: str
    started_at: datetime
    ended_at: datetime
    outcome: str
    reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: int
    legacy_cost_usd: float | None = None


@dataclass(frozen=True)
class EnqueueResult:
    job: Job
    created: bool

    @property
    def job_id(self) -> int:
        return self.job.id

    @property
    def inserted(self) -> bool:
        """Compatibility spelling for callers reporting enqueue outcomes."""
        return self.created


class QueueRepository:
    """Small repository API that keeps queue SQL out of hook code."""

    def __init__(
        self,
        path: Path | str,
        *,
        busy_timeout_ms: int = 5_000,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        clock: Callable[[], datetime] = _utc_now,
        redaction_env: Mapping[str, str] | None = None,
        memory_home: Path | str | None = None,
        sync_usage: bool = True,
    ) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.path = Path(os.path.abspath(Path(path).expanduser()))
        self.memory_home = (
            Path(os.path.abspath(Path(memory_home).expanduser()))
            if memory_home is not None
            else (
                self.path.parent.parent
                if self.path.parent.name == "scripts"
                else self.path.parent
            )
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink():
            raise ValueError("queue parent must not be a symlink")
        self._prepare_private_database_files()
        self.max_attempts = max_attempts
        self._clock = clock
        self._redaction_env = dict(os.environ if redaction_env is None else redaction_env)
        self._connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._migrate()
            self._secure_database_files()
            if sync_usage:
                self._sync_usage_records()
        except BaseException:
            self._connection.close()
            raise

    def _validate_private_file(self, path: Path) -> None:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"queue path must not be a symlink: {path}")
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"queue path must be a regular file: {path}")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise ValueError(f"queue path is not owned by the current user: {path}")
        if info.st_nlink != 1:
            raise ValueError(f"queue path must not be hard-linked: {path}")
        path.chmod(0o600)

    def _prepare_private_database_files(self) -> None:
        candidates = [self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")]
        for candidate in candidates:
            if candidate.exists() or candidate.is_symlink():
                self._validate_private_file(candidate)
        if not self.path.exists():
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            os.close(descriptor)
        self._validate_private_file(self.path)

    def _secure_database_files(self) -> None:
        for candidate in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            if candidate.exists() or candidate.is_symlink():
                self._validate_private_file(candidate)

    def __enter__(self) -> "QueueRepository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _migration_version_observed(self, version: int) -> None:
        """Test seam for synchronizing concurrent migration openers."""

    def _migrate(self) -> None:
        observed_version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        self._migration_version_observed(observed_version)
        if observed_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"queue schema {observed_version} is newer than supported version "
                f"{SCHEMA_VERSION}"
            )
        if observed_version == SCHEMA_VERSION:
            return
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            while True:
                version = self._connection.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"queue schema {version} is newer than supported version "
                        f"{SCHEMA_VERSION}"
                    )
                if version == SCHEMA_VERSION:
                    self._connection.execute("COMMIT")
                    return
                if version == 0:
                    for statement in _QUEUE_V2_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                    self._connection.execute("PRAGMA user_version = 2")
                    continue
                if version == 1:
                    for statement in _VERSION_1_TO_2_STATEMENTS:
                        self._connection.execute(statement)
                    self._connection.execute("PRAGMA user_version = 2")
                    continue
                if version == 2:
                    for statement in _STATUS_SCHEMA_STATEMENTS:
                        self._connection.execute(statement)
                    self._connection.execute("PRAGMA user_version = 3")
                    continue
                raise RuntimeError(f"unsupported queue schema version {version}")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _now(self) -> datetime:
        return _datetime(self._clock())

    def status_run_for_job(self, job_id: int) -> StatusRun:
        row = self._connection.execute(
            "SELECT * FROM status_runs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return status_run_from_row(row, redaction_env=self._redaction_env)

    def _ensure_job_run_unlocked(self, job_id: int) -> StatusRun:
        """Synthesize one authoritative coarse event for a pre-v3 queue job."""
        try:
            return self.status_run_for_job(job_id)
        except KeyError:
            job = self._connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if job is None:
                raise
            started_at = _loaded_time(job["created_at"])
            updated_at = _loaded_time(job["updated_at"])
            if started_at is None or updated_at is None:
                raise ValueError("queue job is missing a required timestamp")
            mapping = _SYNTHESIZED_JOB_STATUS.get(job["status"])
            if mapping is None:
                raise ValueError(f"invalid queue job status: {job['status']!r}")
            state, phase, level, retains_error = mapping
            completed_at = (
                _loaded_time(job["completed_at"])
                if state in {"succeeded", "dead"}
                else None
            )
            error = (
                normalize_persistence_reason(job["last_error"], self._redaction_env)
                if retains_error and job["last_error"] is not None
                else None
            )
            run_id = create_job_run_unlocked(
                self._connection,
                job_id=job["id"],
                kind=job["kind"],
                source_agent=job["source_agent"],
                session_id=job["session_id"],
                project=job["project"],
                now=started_at,
                state=state,
                phase=phase,
                summary=error,
                error=error,
                updated_at=updated_at,
                completed_at=completed_at,
                redaction_env=self._redaction_env,
            )
            details = (
                {"retry_at": _stored_time(job["available_at"])}
                if state == "retrying"
                else None
            )
            append_event_unlocked(
                self._connection,
                run_id,
                phase,
                now=completed_at or updated_at,
                level=level,
                attempt=(job["attempt_count"] or None),
                message=error,
                details=details,
                redaction_env=self._redaction_env,
            )
            return self.status_run_for_job(job_id)

    def status_run_for_operation(self, operation_key: str) -> StatusRun | None:
        row = self._connection.execute(
            "SELECT * FROM status_runs WHERE operation_key = ?", (operation_key,)
        ).fetchone()
        return (
            status_run_from_row(row, redaction_env=self._redaction_env)
            if row is not None
            else None
        )

    def status_events(self, run_id: int) -> tuple[StatusEvent, ...]:
        rows = self._connection.execute(
            "SELECT * FROM status_events WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return tuple(
            status_event_from_row(row, redaction_env=self._redaction_env)
            for row in rows
        )

    def _append_run_event(
        self,
        run: StatusRun,
        phase: str,
        *,
        level: EventLevel,
        provider: ProviderName | None,
        attempt: int | None,
        message: str | None,
        details: Mapping[str, JsonScalar] | None,
    ) -> None:
        transition_run_unlocked(
            self._connection,
            run.id,
            run.state,
            phase,
            now=self._now(),
            summary=run.summary,
            error=run.error,
            completed_at=run.completed_at,
            level=level,
            provider=provider,
            attempt=attempt,
            message=message,
            details=details,
            redaction_env=self._redaction_env,
        )

    def append_job_event(
        self,
        job_id: int,
        owner: str,
        phase: str,
        *,
        expected_attempt_count: int,
        level: EventLevel = "info",
        provider: ProviderName | None = None,
        message: str | None = None,
        details: Mapping[str, JsonScalar] | None = None,
    ) -> None:
        if not owner:
            raise ValueError("owner must not be empty")
        if (
            isinstance(expected_attempt_count, bool)
            or not isinstance(expected_attempt_count, int)
            or expected_attempt_count < 1
        ):
            raise ValueError("expected_attempt_count must be a positive integer")
        if phase not in _JOB_EVENT_PHASES:
            raise ValueError(f"invalid job event phase: {phase!r}")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            job = self._connection.execute(
                """
                SELECT attempt_count FROM jobs
                WHERE id = ? AND status = 'leased' AND lease_owner = ?
                    AND attempt_count = ?
                """,
                (job_id, owner, expected_attempt_count),
            ).fetchone()
            if job is None:
                raise LeaseOwnershipError(f"job {job_id} is not leased by {owner}")
            run = self.status_run_for_job(job_id)
            if run.state != "running":
                raise ValueError(
                    f"leased job {job_id} has incompatible run state {run.state!r}"
                )
            self._append_run_event(
                run,
                phase,
                level=level,
                provider=provider,
                attempt=job["attempt_count"],
                message=message,
                details=details,
            )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def create_operation_run(
        self,
        operation_key: str,
        *,
        kind: RunKind,
        source_agent: SourceAgent,
        session_id: str,
        project: str,
        phase: str = "reserved",
    ) -> StatusRun:
        operation_key, kind, source_agent, session_id, project = (
            validate_operation_identity(
                operation_key,
                kind,
                source_agent,
                session_id,
                project,
                redaction_env=self._redaction_env,
            )
        )
        if phase not in ALLOWED_PHASES:
            raise ValueError(f"invalid status phase: {phase!r}")
        validate_operation_event("queued", phase)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.status_run_for_operation(operation_key)
            if existing is not None:
                identity = (kind, source_agent, session_id, project)
                persisted_identity = (
                    existing.kind,
                    existing.source_agent,
                    existing.session_id,
                    existing.project,
                )
                if persisted_identity != identity:
                    raise ValueError("operation_key is already used by another operation")
                self._connection.execute("COMMIT")
                return existing
            now = self._now()
            run_id = create_operation_run_unlocked(
                self._connection,
                operation_key=operation_key,
                kind=kind,
                source_agent=source_agent,
                session_id=session_id,
                project=project,
                phase=phase,
                now=now,
                redaction_env=self._redaction_env,
            )
            append_event_unlocked(
                self._connection,
                run_id,
                phase,
                now=now,
                redaction_env=self._redaction_env,
            )
            created = self.status_run_for_operation(operation_key)
            if created is None:  # Defensive: insert and readback share a transaction.
                raise RuntimeError("created operation status could not be read back")
            self._connection.execute("COMMIT")
            return created
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def append_operation_event(
        self,
        run_id: int,
        phase: str,
        *,
        level: EventLevel = "info",
        provider: ProviderName | None = None,
        attempt: int | None = None,
        message: str | None = None,
        details: Mapping[str, JsonScalar] | None = None,
    ) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT * FROM status_runs WHERE id = ? AND operation_key IS NOT NULL",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = status_run_from_row(row, redaction_env=self._redaction_env)
            validate_operation_event(run.state, phase)
            self._append_run_event(
                run,
                phase,
                level=level,
                provider=provider,
                attempt=attempt,
                message=message,
                details=details,
            )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def transition_operation_run(
        self,
        run_id: int,
        state: RunState,
        phase: str,
        *,
        summary: str | None = None,
        error: str | None = None,
        level: EventLevel = "info",
        provider: ProviderName | None = None,
        attempt: int | None = None,
        message: str | None = None,
        details: Mapping[str, JsonScalar] | None = None,
    ) -> None:
        now = self._now()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT * FROM status_runs WHERE id = ? AND operation_key IS NOT NULL",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            run = status_run_from_row(row, redaction_env=self._redaction_env)
            validate_operation_transition(run.state, state, phase)
            transition_run_unlocked(
                self._connection,
                run_id,
                state,
                phase,
                now=now,
                summary=summary,
                error=error,
                completed_at=(
                    now if state in {"succeeded", "failed", "dead"} else None
                ),
                level=level,
                provider=provider,
                attempt=attempt,
                message=message,
                details=details,
                redaction_env=self._redaction_env,
            )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    @property
    def queue_id(self) -> str:
        return self._connection.execute(
            "SELECT value FROM queue_metadata WHERE key = 'queue_id'"
        ).fetchone()[0]

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            kind=row["kind"],
            source_agent=row["source_agent"],
            session_id=row["session_id"],
            project=row["project"],
            cwd=row["cwd"],
            trigger=row["trigger"],
            source_path=row["source_path"],
            source_hash=row["source_hash"],
            payload_json=row["payload_json"],
            status=row["status"],
            attempt_count=row["attempt_count"],
            available_at=_loaded_time(row["available_at"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=_loaded_time(row["lease_expires_at"]),
            last_error=row["last_error"],
            created_at=_loaded_time(row["created_at"]),
            updated_at=_loaded_time(row["updated_at"]),
            completed_at=_loaded_time(row["completed_at"]),
        )

    def enqueue_capture(self, session: NormalizedSession) -> EnqueueResult:
        now_dt = self._now()
        now = _stored_time(now_dt)
        payload = json.dumps(
            {
                "timestamp": session.timestamp,
                "turns": [asdict(turn) for turn in session.turns],
                "rendered_context": render_turns(session),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        owns_transaction = not self._connection.in_transaction
        if owns_transaction:
            self._connection.execute("BEGIN IMMEDIATE")
        else:
            self._connection.execute("SAVEPOINT enqueue_capture_status")
        try:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    kind, source_agent, session_id, project, cwd, trigger, source_path,
                    source_hash, payload_json, status, attempt_count, available_at,
                    created_at, updated_at
                ) VALUES ('capture', ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
                """,
                (
                    session.agent,
                    session.session_id,
                    session.project,
                    session.cwd,
                    session.trigger,
                    session.source_path,
                    session.source_hash,
                    payload,
                    now,
                    now,
                    now,
                ),
            )
            created = cursor.rowcount == 1
            row = self._connection.execute(
                """
                SELECT * FROM jobs
                WHERE kind = 'capture' AND source_agent = ? AND session_id = ?
                    AND source_hash = ?
                """,
                (session.agent, session.session_id, session.source_hash),
            ).fetchone()
            if row is None:  # Defensive: insert and readback share this transaction.
                raise RuntimeError("enqueued capture could not be read back")
            if created:
                run_id = create_job_run_unlocked(
                    self._connection,
                    job_id=row["id"],
                    kind=row["kind"],
                    source_agent=row["source_agent"],
                    session_id=row["session_id"],
                    project=row["project"],
                    now=now_dt,
                    redaction_env=self._redaction_env,
                )
                append_event_unlocked(
                    self._connection,
                    run_id,
                    "queued",
                    now=now_dt,
                    redaction_env=self._redaction_env,
                )
            else:
                self._ensure_job_run_unlocked(row["id"])
            result = EnqueueResult(self._job(row), created)
            self._connection.execute(
                "COMMIT" if owns_transaction else "RELEASE enqueue_capture_status"
            )
            return result
        except BaseException:
            if owns_transaction and self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            elif self._connection.in_transaction:
                self._connection.execute("ROLLBACK TO enqueue_capture_status")
                self._connection.execute("RELEASE enqueue_capture_status")
            raise

    def claim_next(
        self,
        owner: str,
        now: datetime | str | int | float,
        lease_seconds: int,
    ) -> Job | None:
        if not owner:
            raise ValueError("owner must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now_dt = _datetime(now)
        now_value = _stored_time(now_dt)
        expires = _stored_time(now_dt + timedelta(seconds=lease_seconds))
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT id FROM jobs
                WHERE status IN ('pending', 'failed') AND available_at <= ?
                ORDER BY available_at, id
                LIMIT 1
                """,
                (now_value,),
            ).fetchone()
            if row is None:
                self._connection.execute("COMMIT")
                return None
            run = self._ensure_job_run_unlocked(row["id"])
            self._connection.execute(
                """
                UPDATE jobs
                SET status = 'leased', attempt_count = attempt_count + 1,
                    lease_owner = ?, lease_expires_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (owner, expires, now_value, row["id"]),
            )
            claimed = self._connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (row["id"],)
            ).fetchone()
            transition_run_unlocked(
                self._connection,
                run.id,
                "running",
                "worker_claimed",
                now=now_dt,
                redaction_env=self._redaction_env,
            )
            self._connection.execute("COMMIT")
            return self._job(claimed)
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def renew(
        self,
        job_id: int,
        owner: str,
        expires_at: datetime | str | int | float,
    ) -> bool:
        cursor = self._connection.execute(
            """
            UPDATE jobs SET lease_expires_at = ?, updated_at = ?
            WHERE id = ? AND status = 'leased' AND lease_owner = ?
            """,
            (_stored_time(expires_at), _stored_time(self._now()), job_id, owner),
        )
        return cursor.rowcount == 1

    def record_attempt(
        self,
        job_id: int,
        result: ProviderResult,
        *,
        legacy_cost_usd: float | None = None,
    ) -> None:
        ended = self._now()
        started = ended - timedelta(milliseconds=max(0, result.elapsed_ms))
        cursor = self._connection.execute(
            """
            INSERT INTO provider_attempts (
                job_id, provider, model, task, started_at, ended_at, outcome,
                reason, input_tokens, output_tokens, elapsed_ms, legacy_cost_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                result.provider,
                result.model,
                result.task.value,
                _stored_time(started),
                _stored_time(ended),
                result.outcome,
                (
                    normalize_persistence_reason(result.reason, self._redaction_env)
                    if result.reason
                    else None
                ),
                result.input_tokens,
                result.output_tokens,
                max(0, result.elapsed_ms),
                legacy_cost_usd if result.provider == "claude" else None,
            ),
        )
        self._append_attempt_usage(cursor.lastrowid)

    def _usage_row(self, attempt_id: int) -> sqlite3.Row:
        row = self._connection.execute(
            """
            SELECT a.*, j.source_agent
            FROM provider_attempts AS a
            JOIN jobs AS j ON j.id = a.job_id
            WHERE a.id = ?
            """,
            (attempt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return row

    def _fallback_reason_for(self, row: sqlite3.Row) -> str | None:
        if row["provider"] != "claude":
            return None
        prior = self._connection.execute(
            """
            SELECT outcome, reason FROM provider_attempts
            WHERE job_id = ? AND id < ? AND provider = 'codex'
            ORDER BY id DESC LIMIT 1
            """,
            (row["job_id"], row["id"]),
        ).fetchone()
        if prior is None:
            return None
        reason = prior["reason"] or prior["outcome"]
        return f"codex:{prior['outcome']}:{reason}"

    def _append_attempt_usage(self, attempt_id: int) -> None:
        """Best-effort JSONL projection after the authoritative DB commit."""
        row = self._usage_row(attempt_id)
        try:
            append_usage_record(
                self.memory_home,
                UsageRecord(
                    provider=row["provider"],
                    model=row["model"],
                    task=row["task"],
                    source_agent=row["source_agent"],
                    outcome=row["outcome"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    elapsed_ms=row["elapsed_ms"],
                    timestamp=row["ended_at"],
                    job_id=row["job_id"],
                    fallback_reason=self._fallback_reason_for(row),
                    reason=row["reason"],
                    provider_attempt_id=row["id"],
                    queue_id=self.queue_id,
                    legacy_cost_usd=row["legacy_cost_usd"],
                ),
                env=self._redaction_env,
            )
        except (OSError, ValueError):
            # Observability is recoverable; it must not roll back source-of-truth
            # provider_attempts. The next repository open retries the projection.
            return

    def _sync_usage_records(self) -> None:
        try:
            recover_usage_log(self.memory_home)
            logged = logged_provider_attempt_ids(self.memory_home)
        except UnsafeUsagePathError:
            raise
        except (OSError, ValueError):
            logged = set()
        rows = self._connection.execute(
            "SELECT id FROM provider_attempts ORDER BY id"
        ).fetchall()
        for row in rows:
            if (self.queue_id, row["id"]) not in logged:
                self._append_attempt_usage(row["id"])

    def sync_usage_records(self) -> None:
        """Recover and project usage after the caller owns worker singleton."""
        self._sync_usage_records()

    def complete(
        self,
        job_id: int,
        owner: str,
        *,
        summary: str | None = None,
    ) -> None:
        now_dt = self._now()
        now = _stored_time(now_dt)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            ownership = self._connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (job_id, owner),
            ).fetchone()
            if ownership is None:
                raise LeaseOwnershipError(f"job {job_id} is not leased by {owner}")
            run = self._ensure_job_run_unlocked(job_id)
            cursor = self._connection.execute(
                """
                UPDATE jobs
                SET status = 'succeeded', lease_owner = NULL, lease_expires_at = NULL,
                    last_error = NULL, updated_at = ?, completed_at = ?
                WHERE id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (now, now, job_id, owner),
            )
            if cursor.rowcount != 1:
                raise LeaseOwnershipError(f"job {job_id} is not leased by {owner}")
            transition_run_unlocked(
                self._connection,
                run.id,
                "succeeded",
                "succeeded",
                now=now_dt,
                summary=summary,
                completed_at=now_dt,
                redaction_env=self._redaction_env,
            )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def retry(
        self,
        job_id: int,
        owner: str,
        error: str,
        available_at: datetime | str | int | float,
    ) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT attempt_count FROM jobs
                WHERE id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (job_id, owner),
            ).fetchone()
            if row is None:
                raise LeaseOwnershipError(f"job {job_id} is not leased by {owner}")
            run = self._ensure_job_run_unlocked(job_id)
            now_dt = self._now()
            now = _stored_time(now_dt)
            dead = row["attempt_count"] >= self.max_attempts
            normalized_error = normalize_persistence_reason(error, self._redaction_env)
            retry_at = _stored_time(available_at)
            cursor = self._connection.execute(
                """
                UPDATE jobs
                SET status = ?, available_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL, last_error = ?, updated_at = ?, completed_at = ?
                WHERE id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (
                    "dead" if dead else "failed",
                    retry_at,
                    normalized_error,
                    now,
                    now if dead else None,
                    job_id,
                    owner,
                ),
            )
            if cursor.rowcount != 1:
                raise LeaseOwnershipError(f"job {job_id} is not leased by {owner}")
            transition_run_unlocked(
                self._connection,
                run.id,
                "dead" if dead else "retrying",
                "dead" if dead else "retry_wait",
                now=now_dt,
                summary=normalized_error,
                error=normalized_error,
                completed_at=now_dt if dead else None,
                level="error" if dead else "warning",
                message=normalized_error,
                details=None if dead else {"retry_at": retry_at},
                redaction_env=self._redaction_env,
            )
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def recover_stale(self, now: datetime | str | int | float) -> int:
        now_dt = _datetime(now)
        now_value = _stored_time(now_dt)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            stale = self._connection.execute(
                """
                SELECT id, attempt_count FROM jobs
                WHERE status = 'leased' AND lease_expires_at <= ?
                ORDER BY id
                """,
                (now_value,),
            ).fetchall()
            runs = {
                job["id"]: self._ensure_job_run_unlocked(job["id"])
                for job in stale
            }
            cursor = self._connection.execute(
                """
                UPDATE jobs
                SET status = CASE WHEN attempt_count >= ? THEN 'dead' ELSE 'failed' END,
                    available_at = ?, lease_owner = NULL, lease_expires_at = NULL,
                    last_error = 'worker lease expired', updated_at = ?,
                    completed_at = CASE WHEN attempt_count >= ? THEN ? ELSE NULL END
                WHERE status = 'leased' AND lease_expires_at <= ?
                """,
                (
                    self.max_attempts,
                    now_value,
                    now_value,
                    self.max_attempts,
                    now_value,
                    now_value,
                ),
            )
            for job in stale:
                dead = job["attempt_count"] >= self.max_attempts
                run = runs[job["id"]]
                transition_run_unlocked(
                    self._connection,
                    run.id,
                    "dead" if dead else "retrying",
                    "dead" if dead else "recovery_pending",
                    now=now_dt,
                    summary="worker lease expired",
                    error="worker lease expired",
                    completed_at=now_dt if dead else None,
                    level="error" if dead else "warning",
                    message="worker lease expired",
                    redaction_env=self._redaction_env,
                )
            self._connection.execute("COMMIT")
            return cursor.rowcount
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def get_job(self, job_id: int) -> Job:
        row = self._connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job(row)

    def count_jobs(self) -> int:
        return self._connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def next_available_at(self) -> datetime | None:
        row = self._connection.execute(
            """
            SELECT MIN(available_at) FROM jobs
            WHERE status IN ('pending', 'failed')
            """
        ).fetchone()
        return _loaded_time(row[0]) if row and row[0] is not None else None

    def next_wake_at(self) -> datetime | None:
        """Return the next runnable time or active lease expiry."""
        row = self._connection.execute(
            """
            SELECT MIN(wake_at) FROM (
                SELECT available_at AS wake_at FROM jobs
                WHERE status IN ('pending', 'failed')
                UNION ALL
                SELECT lease_expires_at AS wake_at FROM jobs
                WHERE status = 'leased' AND lease_expires_at IS NOT NULL
            )
            """
        ).fetchone()
        return _loaded_time(row[0]) if row and row[0] is not None else None

    def release_worker_lock_if_idle(self, release: Callable[[], None]) -> bool:
        """Release a worker lock only while enqueues are serialized behind us."""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            active = self._connection.execute(
                """
                SELECT 1 FROM jobs
                WHERE status IN ('pending', 'failed', 'leased')
                LIMIT 1
                """
            ).fetchone()
            if active is not None:
                self._connection.execute("COMMIT")
                return False
            release()
            self._connection.execute("COMMIT")
            return True
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _valid_reservation_component(value: str) -> bool:
        return len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )

    @classmethod
    def _validate_auto_compile_read(
        cls, observation: AutoCompileContentRead
    ) -> tuple[AutoCompileReadStatus, str | None, tuple[str, ...] | None]:
        if not isinstance(observation, tuple) or len(observation) not in {2, 3}:
            raise ValueError("invalid auto-compile content observation")
        try:
            status, fingerprint = observation[:2]
            markers = observation[2] if len(observation) == 3 else None
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid auto-compile content observation") from exc
        if status not in {"unreadable", "covered", "uncompiled"}:
            raise ValueError("invalid auto-compile content status")
        if status == "uncompiled":
            if not isinstance(fingerprint, str) or not cls._valid_reservation_component(
                fingerprint
            ):
                raise ValueError("uncompiled observation requires a fingerprint")
        elif fingerprint is not None:
            raise ValueError("non-uncompiled observation cannot have a fingerprint")
        if markers is not None and (
            not isinstance(markers, tuple)
            or not all(isinstance(marker, str) for marker in markers)
        ):
            raise ValueError("invalid auto-compile marker observation")
        return status, fingerprint, markers

    def _active_work_unlocked(self) -> bool:
        return self._connection.execute(
            """
            SELECT 1 FROM jobs
            WHERE status IN ('pending', 'failed', 'leased')
            LIMIT 1
            """
        ).fetchone() is not None

    @staticmethod
    def _auto_compile_operation_key(log_name: str, fingerprint: str) -> str:
        return f"auto-compile:{log_name}:{fingerprint}"

    def _create_auto_compile_run_unlocked(
        self,
        log_name: str,
        fingerprint: str,
        now: datetime,
        execution_token: str | None = None,
    ) -> int:
        base_key = self._auto_compile_operation_key(log_name, fingerprint)
        operation_key = base_key
        row = self._connection.execute(
            "SELECT * FROM status_runs WHERE operation_key = ?", (operation_key,)
        ).fetchone()
        if row is not None:
            existing = status_run_from_row(row, redaction_env=self._redaction_env)
            if existing.state not in {"succeeded", "failed", "dead"}:
                return existing.id
            suffix = (execution_token or secrets.token_hex(8))[:16]
            operation_key = f"{base_key}:{suffix}"
            counter = 1
            while self._connection.execute(
                "SELECT 1 FROM status_runs WHERE operation_key = ?", (operation_key,)
            ).fetchone() is not None:
                operation_key = f"{base_key}:{suffix}-{counter}"
                counter += 1
        project = self.memory_home.name
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,256}", project):
            project = "global"
        run_id = create_operation_run_unlocked(
            self._connection,
            operation_key=operation_key,
            kind="compile",
            source_agent="system",
            session_id="",
            project=project,
            phase="reserved",
            now=now,
            redaction_env=self._redaction_env,
        )
        append_event_unlocked(
            self._connection,
            run_id,
            "reserved",
            now=now,
            redaction_env=self._redaction_env,
        )
        return run_id

    def _ensure_auto_compile_links_unlocked(
        self, reservation: dict[str, object], now: datetime
    ) -> bool:
        changed = False
        fingerprint = reservation.get("fingerprint")
        log_name = reservation.get("log_name")
        if isinstance(fingerprint, str) and isinstance(log_name, str):
            run = self._auto_compile_run_unlocked(reservation)
            if run is None or run.state in {"succeeded", "failed", "dead"}:
                reservation["status_run_id"] = self._create_auto_compile_run_unlocked(
                    log_name,
                    fingerprint,
                    now,
                    execution_token=str(reservation.get("token", "legacy")),
                )
                changed = True
        pending = reservation.get("pending_fingerprint")
        pending_log = reservation.get("pending_log_name")
        if isinstance(pending, str) and isinstance(pending_log, str):
            run = self._auto_compile_run_unlocked(reservation, pending=True)
            if run is None or run.state in {"succeeded", "failed", "dead"}:
                reservation["pending_status_run_id"] = (
                    self._create_auto_compile_run_unlocked(
                        pending_log,
                        pending,
                        now,
                        execution_token=str(
                            reservation.get("watcher_token", "legacy-pending")
                        ),
                    )
                )
                changed = True
        return changed

    def _delete_auto_compile_reservation_unlocked(
        self,
        reservation: Mapping[str, object],
        *,
        now: datetime,
        active_succeeded: bool,
        reason: str,
    ) -> None:
        active = self._auto_compile_run_unlocked(reservation)
        pending = self._auto_compile_run_unlocked(reservation, pending=True)
        for run, succeeded in ((active, active_succeeded), (pending, False)):
            if run is None or run.state in {"succeeded", "failed", "dead"}:
                continue
            if succeeded:
                if run.state in {"queued", "retrying"}:
                    self._transition_auto_compile_run_unlocked(
                        {"status_run_id": run.id},
                        "running",
                        "generation_recovered",
                        now=now,
                        summary="Recovered completed automatic compile",
                    )
                self._transition_auto_compile_run_unlocked(
                    {"status_run_id": run.id},
                    "succeeded",
                    "succeeded",
                    now=now,
                    summary=reason,
                )
            else:
                transition_run_unlocked(
                    self._connection,
                    run.id,
                    "failed",
                    "failed",
                    now=now,
                    summary=reason,
                    completed_at=now,
                    redaction_env=self._redaction_env,
                )
        self._connection.execute(
            "DELETE FROM queue_metadata WHERE key = ?",
            (AUTO_COMPILE_RESERVATION_KEY,),
        )

    def _auto_compile_run_unlocked(
        self, reservation: Mapping[str, object], *, pending: bool = False
    ) -> StatusRun | None:
        key = "pending_status_run_id" if pending else "status_run_id"
        run_id = reservation.get(key)
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
            return None
        row = self._connection.execute(
            "SELECT * FROM status_runs WHERE id = ? AND operation_key IS NOT NULL",
            (run_id,),
        ).fetchone()
        return (
            status_run_from_row(row, redaction_env=self._redaction_env)
            if row is not None
            else None
        )

    def _transition_auto_compile_run_unlocked(
        self,
        reservation: Mapping[str, object],
        state: RunState,
        phase: str,
        *,
        now: datetime,
        summary: str | None = None,
        error: str | None = None,
        level: EventLevel = "info",
        message: str | None = None,
        details: Mapping[str, JsonScalar] | None = None,
    ) -> None:
        run = self._auto_compile_run_unlocked(reservation)
        if run is None or run.state in {"succeeded", "failed", "dead"}:
            return
        if run.state == state:
            validate_operation_event(run.state, phase)
        else:
            validate_operation_transition(run.state, state, phase)
        transition_run_unlocked(
            self._connection,
            run.id,
            state,
            phase,
            now=now,
            summary=summary,
            error=error,
            completed_at=now if state in {"succeeded", "failed", "dead"} else None,
            level=level,
            message=message,
            details=details,
            redaction_env=self._redaction_env,
        )

    def active_auto_compile_status_run(
        self, run_id: int, *, now: datetime | str | int | float
    ) -> StatusRun | None:
        row = self._connection.execute(
            "SELECT value FROM queue_metadata WHERE key = ?",
            (AUTO_COMPILE_RESERVATION_KEY,),
        ).fetchone()
        if row is None:
            return None
        try:
            reservation = json.loads(row["value"])
            unexpired = _datetime(reservation["expires_at"]) > _datetime(now)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        run = self._auto_compile_run_unlocked(reservation)
        return run if unexpired and run is not None and run.id == run_id else None

    def record_active_auto_compile_phase(
        self,
        run_id: int,
        phase: str,
        *,
        details: Mapping[str, JsonScalar] | None = None,
    ) -> bool:
        """Record a child phase only while its exact reservation remains active."""
        now = self._now()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            if row is None:
                self._connection.execute("COMMIT")
                return False
            try:
                reservation = json.loads(row["value"])
                unexpired = _datetime(reservation["expires_at"]) > now
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                unexpired = False
                reservation = {}
            run = self._auto_compile_run_unlocked(reservation)
            if not unexpired or run is None or run.id != run_id:
                self._connection.execute("COMMIT")
                return False
            if phase == "staging_started" and run.state == "queued":
                self._transition_auto_compile_run_unlocked(
                    reservation, "running", phase, now=now, details=details
                )
            elif phase == "staging_started" and run.state == "retrying":
                self._transition_auto_compile_run_unlocked(
                    reservation,
                    "running",
                    "generation_recovered",
                    now=now,
                    summary="Retrying automatic compile",
                )
                self._transition_auto_compile_run_unlocked(
                    reservation, "running", phase, now=now, details=details
                )
            elif run.state == "running":
                if run.phase == phase and phase == "staging_started":
                    self._connection.execute("COMMIT")
                    return True
                self._transition_auto_compile_run_unlocked(
                    reservation, "running", phase, now=now, details=details
                )
            else:
                self._connection.execute("COMMIT")
                return False
            self._connection.execute("COMMIT")
            return True
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def reserve_auto_compile(
        self,
        token: str,
        fingerprint: str,
        *,
        log_name: str | None = None,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
    ) -> bool:
        """Reserve one idle-queue compile lease for a content fingerprint."""
        if not self._valid_reservation_component(token):
            raise ValueError("auto-compile token must be a lowercase SHA-256 value")
        if not self._valid_reservation_component(fingerprint):
            raise ValueError("auto-compile fingerprint must be a lowercase SHA-256 value")
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if expires_dt <= now_dt:
            raise ValueError("auto-compile reservation expiry must be in the future")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if self._active_work_unlocked():
                self._connection.execute("COMMIT")
                return False
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            if row is not None:
                try:
                    existing = json.loads(row["value"])
                    existing_expiry = _datetime(existing["expires_at"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    existing_expiry = now_dt
                if existing_expiry > now_dt:
                    if (
                        existing.get("fingerprint") != fingerprint
                        and existing.get("pending_fingerprint") != fingerprint
                    ):
                        existing["pending_fingerprint"] = fingerprint
                        if log_name is not None:
                            existing["pending_log_name"] = log_name
                        self._connection.execute(
                            "UPDATE queue_metadata SET value = ? WHERE key = ?",
                            (
                                json.dumps(
                                    existing,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                AUTO_COMPILE_RESERVATION_KEY,
                            ),
                        )
                    self._connection.execute("COMMIT")
                    return False
            reservation = json.dumps(
                {
                    "token": token,
                    "fingerprint": fingerprint,
                    "expires_at": _stored_time(expires_dt),
                    **({"log_name": log_name} if log_name is not None else {}),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            self._connection.execute(
                """
                INSERT INTO queue_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (AUTO_COMPILE_RESERVATION_KEY, reservation),
            )
            self._connection.execute("COMMIT")
            return True
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def request_auto_compile(
        self,
        token: str,
        fingerprint: str,
        *,
        log_name: str,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
    ) -> str | None:
        """Atomically become the owner or the sole live watcher."""
        if not self._valid_reservation_component(token):
            raise ValueError("auto-compile token must be a lowercase SHA-256 value")
        if not self._valid_reservation_component(fingerprint):
            raise ValueError("auto-compile fingerprint must be lowercase SHA-256")
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if expires_dt <= now_dt:
            raise ValueError("auto-compile reservation expiry must be in the future")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if self._active_work_unlocked():
                self._connection.execute("COMMIT")
                return None
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation: dict[str, object] = {}
            owner_live = False
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                    owner_live = _datetime(reservation["expires_at"]) > now_dt
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    reservation = {}
            if not owner_live:
                reservation = {
                    "token": token,
                    "fingerprint": fingerprint,
                    "log_name": log_name,
                    "required_marker_prefix": [],
                    "expires_at": _stored_time(expires_dt),
                    "status_run_id": self._create_auto_compile_run_unlocked(
                        log_name, fingerprint, now_dt, execution_token=token
                    ),
                }
                role = "owner"
            else:
                reservation["pending_status_run_id"] = (
                    reservation.get("status_run_id")
                    if reservation.get("fingerprint") == fingerprint
                    else self._create_auto_compile_run_unlocked(
                        log_name, fingerprint, now_dt, execution_token=token
                    )
                )
                reservation["pending_fingerprint"] = fingerprint
                reservation["pending_log_name"] = log_name
                reservation["pending_required_marker_prefix"] = []
                try:
                    watcher_live = (
                        _datetime(reservation["watcher_expires_at"]) > now_dt
                        and isinstance(reservation.get("watcher_token"), str)
                    )
                except (KeyError, TypeError, ValueError):
                    watcher_live = False
                if watcher_live:
                    self._connection.execute(
                        "UPDATE queue_metadata SET value = ? WHERE key = ?",
                        (
                            json.dumps(
                                reservation, sort_keys=True, separators=(",", ":")
                            ),
                            AUTO_COMPILE_RESERVATION_KEY,
                        ),
                    )
                    self._connection.execute("COMMIT")
                    return None
                reservation["watcher_token"] = token
                reservation["watcher_expires_at"] = _stored_time(expires_dt)
                role = "watcher"
            self._connection.execute(
                """
                INSERT INTO queue_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (
                    AUTO_COMPILE_RESERVATION_KEY,
                    json.dumps(reservation, sort_keys=True, separators=(",", ":")),
                ),
            )
            self._connection.execute("COMMIT")
            return role
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def schedule_auto_compile(
        self,
        owner_token: str,
        watchdog_token: str,
        fingerprint: str,
        *,
        log_name: str,
        required_marker_prefix: tuple[str, ...] = (),
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
        launch_roles: Callable[[tuple[str, ...], str | None], None] | None = None,
    ) -> tuple[str, ...]:
        """Atomically provision an owner and exactly one takeover watchdog."""
        for label, value in (
            ("owner token", owner_token),
            ("watchdog token", watchdog_token),
            ("fingerprint", fingerprint),
        ):
            if not self._valid_reservation_component(value):
                raise ValueError(f"auto-compile {label} must be lowercase SHA-256")
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if expires_dt <= now_dt:
            raise ValueError("auto-compile reservation expiry must be in the future")
        if not all(isinstance(marker, str) for marker in required_marker_prefix):
            raise ValueError("invalid auto-compile marker prefix")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if self._active_work_unlocked():
                self._connection.execute("COMMIT")
                return ()
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation: dict[str, object] = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            if reservation:
                self._ensure_auto_compile_links_unlocked(reservation, now_dt)
            status = reservation.get("status")
            predecessor_token: str | None = None
            if not reservation or status == "failed":
                status_run_id = self._create_auto_compile_run_unlocked(
                    log_name,
                    fingerprint,
                    now_dt,
                    execution_token=owner_token,
                )
                reservation = {
                    "token": owner_token,
                    "fingerprint": fingerprint,
                    "log_name": log_name,
                    "expires_at": _stored_time(expires_dt),
                    "watcher_token": watchdog_token,
                    "watcher_expires_at": _stored_time(expires_dt),
                    "required_marker_prefix": list(required_marker_prefix),
                    "status_run_id": status_run_id,
                }
                roles = ("owner", "watchdog")
            else:
                if reservation.get("fingerprint") == fingerprint:
                    pending_status_run_id = reservation.get("status_run_id")
                else:
                    pending_status_run_id = self._create_auto_compile_run_unlocked(
                        log_name,
                        fingerprint,
                        now_dt,
                        execution_token=watchdog_token,
                    )
                prior_pending = self._auto_compile_run_unlocked(
                    reservation, pending=True
                )
                if (
                    prior_pending is not None
                    and prior_pending.id != pending_status_run_id
                    and prior_pending.state not in {"succeeded", "failed", "dead"}
                ):
                    transition_run_unlocked(
                        self._connection,
                        prior_pending.id,
                        "failed",
                        "failed",
                        now=now_dt,
                        summary="Superseded by newer uncompiled content",
                        completed_at=now_dt,
                        redaction_env=self._redaction_env,
                    )
                reservation["pending_fingerprint"] = fingerprint
                reservation["pending_log_name"] = log_name
                reservation["pending_required_marker_prefix"] = list(
                    required_marker_prefix
                )
                reservation["pending_status_run_id"] = pending_status_run_id
                try:
                    watcher_live = (
                        _datetime(reservation["watcher_expires_at"]) > now_dt
                        and isinstance(reservation.get("watcher_token"), str)
                    )
                except (KeyError, TypeError, ValueError):
                    watcher_live = False
                try:
                    contender_live = (
                        _datetime(reservation["contender_expires_at"]) > now_dt
                        and isinstance(reservation.get("contender_token"), str)
                    )
                except (KeyError, TypeError, ValueError):
                    contender_live = False
                if (
                    watcher_live
                    and status in {"queue_wait", "retry_wait", "read_wait"}
                    and not contender_live
                ):
                    reservation["contender_token"] = watchdog_token
                    reservation["contender_predecessor_token"] = reservation[
                        "watcher_token"
                    ]
                    predecessor_token = str(reservation["watcher_token"])
                    reservation["contender_expires_at"] = _stored_time(expires_dt)
                    roles = ("contender",)
                elif watcher_live:
                    roles = ()
                else:
                    reservation.pop("contender_token", None)
                    reservation.pop("contender_predecessor_token", None)
                    reservation.pop("contender_expires_at", None)
                    reservation["watcher_token"] = watchdog_token
                    reservation["watcher_expires_at"] = _stored_time(expires_dt)
                    roles = ("watchdog",)
            self._connection.execute(
                """
                INSERT INTO queue_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (
                    AUTO_COMPILE_RESERVATION_KEY,
                    json.dumps(reservation, sort_keys=True, separators=(",", ":")),
                ),
            )
            if roles and launch_roles is not None:
                launch_roles(roles, predecessor_token)
            self._connection.execute("COMMIT")
            return roles
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def rollback_auto_compile_schedule(
        self,
        owner_token: str,
        watchdog_token: str,
        *,
        now: datetime | str | int | float,
    ) -> bool:
        """Roll back an unstarted pair without erasing a concurrent generation."""
        now_dt = _datetime(now)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            matched = (
                reservation.get("token") == owner_token
                and reservation.get("watcher_token") == watchdog_token
            )
            has_pending = isinstance(reservation.get("pending_fingerprint"), str)
            if matched and has_pending:
                reservation.pop("watcher_token", None)
                reservation.pop("watcher_expires_at", None)
                reservation["expires_at"] = _stored_time(now_dt)
                reservation["status"] = "spawn_failed"
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
            elif matched:
                self._delete_auto_compile_reservation_unlocked(
                    reservation,
                    now=now_dt,
                    active_succeeded=False,
                    reason="Automatic compile scheduling rolled back",
                )
            self._connection.execute("COMMIT")
            return matched
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def clear_auto_compile_watcher(self, token: str) -> bool:
        """Clear only the matching watcher role after its process fails to spawn."""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            matched = reservation.get("watcher_token") == token
            if matched:
                reservation.pop("watcher_token", None)
                reservation.pop("watcher_expires_at", None)
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
            self._connection.execute("COMMIT")
            return matched
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def auto_compile_contender_predecessor(self, token: str) -> str | None:
        """Return the registered predecessor for one exact contender token."""
        row = self._connection.execute(
            "SELECT value FROM queue_metadata WHERE key = ?",
            (AUTO_COMPILE_RESERVATION_KEY,),
        ).fetchone()
        if row is None:
            return None
        try:
            reservation = json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            return None
        predecessor = reservation.get("contender_predecessor_token")
        if (
            reservation.get("contender_token") != token
            or not isinstance(predecessor, str)
            or not self._valid_reservation_component(predecessor)
        ):
            return None
        return predecessor

    def auto_compile_role_registered(
        self,
        token: str,
        role: str,
        *,
        predecessor_token: str | None,
    ) -> bool:
        """Return whether one pre-commit child role is now durably registered."""
        row = self._connection.execute(
            "SELECT value FROM queue_metadata WHERE key = ?",
            (AUTO_COMPILE_RESERVATION_KEY,),
        ).fetchone()
        if row is None:
            return False
        try:
            reservation = json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            return False
        if role == "owner":
            return reservation.get("token") == token
        if role == "watchdog":
            return reservation.get("watcher_token") == token
        if role == "contender":
            return (
                reservation.get("contender_token") == token
                and reservation.get("contender_predecessor_token")
                == predecessor_token
            )
        return False

    def bootstrap_auto_compile_watchdog(
        self,
        watchdog_token: str,
        owner_token: str,
        fingerprint: str,
        *,
        log_name: str,
        required_marker_prefix: tuple[str, ...],
        current_content: Callable[[], AutoCompileContentRead],
        now: datetime | str | int | float,
        watcher_expires_at: datetime | str | int | float,
    ) -> str:
        """Recover a rolled-back genesis registration from its spawned watchdog."""
        for value in (watchdog_token, owner_token, fingerprint):
            if not self._valid_reservation_component(value):
                raise ValueError("invalid auto-compile bootstrap identity")
        if Path(log_name).name != log_name or not all(
            isinstance(marker, str) for marker in required_marker_prefix
        ):
            raise ValueError("invalid auto-compile bootstrap source")
        now_dt = _datetime(now)
        watcher_expiry = _datetime(watcher_expires_at)
        if watcher_expiry <= now_dt:
            raise ValueError("bootstrap watcher expiry must be in the future")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if self._active_work_unlocked():
                self._connection.execute("COMMIT")
                return "rejected"
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
                exact = (
                    reservation.get("token") == owner_token
                    and reservation.get("watcher_token") == watchdog_token
                    and reservation.get("fingerprint") == fingerprint
                    and reservation.get("log_name") == log_name
                    and reservation.get("required_marker_prefix")
                    == list(required_marker_prefix)
                )
                self._connection.execute("COMMIT")
                return "registered" if exact else "rejected"
            status, observed_fingerprint, observed_markers = (
                self._validate_auto_compile_read(current_content())
            )
            if (
                status != "uncompiled"
                or observed_fingerprint != fingerprint
                or observed_markers != required_marker_prefix
            ):
                self._connection.execute("COMMIT")
                return "rejected"
            reservation = {
                "token": owner_token,
                "fingerprint": fingerprint,
                "log_name": log_name,
                "required_marker_prefix": list(required_marker_prefix),
                "expires_at": _stored_time(now_dt),
                "watcher_token": watchdog_token,
                "watcher_expires_at": _stored_time(watcher_expiry),
                "status_run_id": self._create_auto_compile_run_unlocked(
                    log_name, fingerprint, now_dt
                ),
            }
            self._connection.execute(
                "INSERT INTO queue_metadata(key, value) VALUES (?, ?)",
                (
                    AUTO_COMPILE_RESERVATION_KEY,
                    json.dumps(reservation, sort_keys=True, separators=(",", ":")),
                ),
            )
            self._connection.execute("COMMIT")
            return "bootstrapped"
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def clear_auto_compile_contender(self, token: str) -> bool:
        """Clear only the matching contender after its process fails to spawn."""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            matched = reservation.get("contender_token") == token
            if matched:
                reservation.pop("contender_token", None)
                reservation.pop("contender_predecessor_token", None)
                reservation.pop("contender_expires_at", None)
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
            self._connection.execute("COMMIT")
            return matched
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def fail_auto_compile_owner_spawn(
        self,
        token: str,
        fingerprint: str,
        *,
        now: datetime | str | int | float,
    ) -> bool:
        """Release a failed owner spawn without erasing a live watcher."""
        now_dt = _datetime(now)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            matched = (
                reservation.get("token") == token
                and reservation.get("fingerprint") == fingerprint
            )
            if matched and isinstance(reservation.get("watcher_token"), str):
                reservation["expires_at"] = _stored_time(now_dt)
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
            elif matched:
                self._delete_auto_compile_reservation_unlocked(
                    reservation,
                    now=now_dt,
                    active_succeeded=False,
                    reason="Automatic compile owner failed to start",
                )
            self._connection.execute("COMMIT")
            return matched
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def poll_auto_compile_watcher(
        self,
        token: str,
        successor_token: str,
        observe: Callable[
            [Mapping[str, object]],
            tuple[AutoCompileReadStatus, Mapping[str, object] | None],
        ],
        launch_successor: Callable[[str], None],
        *,
        predecessor_token: str | None,
        registration_required: bool = False,
        compiler_lock_held: bool = False,
        now: datetime | str | int | float,
        watcher_expires_at: datetime | str | int | float,
        owner_expires_at: datetime | str | int | float,
    ) -> tuple[str, str | None]:
        """Heartbeat a watcher or atomically promote it after owner expiry."""
        if not self._valid_reservation_component(successor_token):
            raise ValueError("successor token must be a lowercase SHA-256 value")
        if predecessor_token is not None and not self._valid_reservation_component(
            predecessor_token
        ):
            raise ValueError("predecessor token must be a lowercase SHA-256 value")
        now_dt = _datetime(now)
        watcher_expiry = _datetime(watcher_expires_at)
        owner_expiry = _datetime(owner_expires_at)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            if row is None:
                self._connection.execute("COMMIT")
                return "done", None
            try:
                reservation = json.loads(row["value"])
                current_owner_expiry = _datetime(reservation["expires_at"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self._connection.execute("COMMIT")
                return "done", None
            registered_watcher = reservation.get("watcher_token")
            if registered_watcher != token:
                registered_contender = reservation.get("contender_token")
                if registration_required and registered_contender != token:
                    self._connection.execute("COMMIT")
                    return "done", None
                recovering_watcher = (
                    predecessor_token is not None
                    and registered_watcher == predecessor_token
                )
                recovering_contender = (
                    predecessor_token is not None
                    and registered_contender == predecessor_token
                )
                if recovering_contender:
                    try:
                        contender_expiry = _datetime(
                            reservation["contender_expires_at"]
                        )
                    except (KeyError, TypeError, ValueError):
                        contender_expiry = now_dt
                    if contender_expiry > now_dt:
                        self._connection.execute("COMMIT")
                        return "wait", None
                    reservation["contender_token"] = token
                    reservation["contender_expires_at"] = _stored_time(
                        watcher_expiry
                    )
                    registered_contender = token
                    recovering_watcher = (
                        reservation.get("contender_predecessor_token")
                        == registered_watcher
                    )
                if not recovering_watcher or registered_contender not in {None, token}:
                    self._connection.execute("COMMIT")
                    return "done", None
                try:
                    predecessor_expiry = _datetime(reservation["watcher_expires_at"])
                except (KeyError, TypeError, ValueError):
                    predecessor_expiry = now_dt
                if predecessor_expiry > now_dt:
                    if registered_contender == token:
                        reservation["contender_expires_at"] = _stored_time(
                            watcher_expiry
                        )
                        self._connection.execute(
                            "UPDATE queue_metadata SET value = ? WHERE key = ?",
                            (
                                json.dumps(
                                    reservation,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                                AUTO_COMPILE_RESERVATION_KEY,
                            ),
                        )
                    self._connection.execute("COMMIT")
                    return "wait", None
                reservation["watcher_token"] = token
                reservation["watcher_expires_at"] = _stored_time(watcher_expiry)
                reservation.pop("contender_token", None)
                reservation.pop("contender_predecessor_token", None)
                reservation.pop("contender_expires_at", None)
            if reservation.get("status") == "failed":
                reservation.pop("watcher_token", None)
                reservation.pop("watcher_expires_at", None)
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return "done", None
            if self._active_work_unlocked():
                reservation["watcher_expires_at"] = _stored_time(watcher_expiry)
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return "wait", None
            if current_owner_expiry > now_dt or compiler_lock_held:
                reservation["watcher_expires_at"] = _stored_time(watcher_expiry)
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return "wait", None
            observed_status, observed = observe(reservation)
            if observed_status == "unreadable":
                reservation["watcher_expires_at"] = _stored_time(watcher_expiry)
                reservation["status"] = "read_wait"
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return "wait", None
            if observed_status == "covered":
                completed_run = self._auto_compile_run_unlocked(reservation)
                if completed_run is not None and completed_run.state in {
                    "queued",
                    "retrying",
                }:
                    self._transition_auto_compile_run_unlocked(
                        reservation,
                        "running",
                        "generation_recovered",
                        now=now_dt,
                        summary="Recovered completed automatic compile",
                    )
                self._transition_auto_compile_run_unlocked(
                    reservation,
                    "succeeded",
                    "succeeded",
                    now=now_dt,
                    summary=f"Compiled {reservation.get('log_name', 'daily log')}",
                )
                self._delete_auto_compile_reservation_unlocked(
                    reservation,
                    now=now_dt,
                    active_succeeded=True,
                    reason=f"Compiled {reservation.get('log_name', 'daily log')}",
                )
                self._connection.execute("COMMIT")
                return "done", None
            if observed_status != "uncompiled" or observed is None:
                raise ValueError("invalid watcher content observation")
            observed_fingerprint = observed.get("fingerprint")
            observed_log_name = observed.get("log_name")
            observed_prefix = observed.get("required_marker_prefix")
            if (
                not isinstance(observed_fingerprint, str)
                or not self._valid_reservation_component(observed_fingerprint)
                or not isinstance(observed_log_name, str)
                or Path(observed_log_name).name != observed_log_name
                or (
                    observed_prefix is not None
                    and (
                        not isinstance(observed_prefix, list)
                        or not all(
                            isinstance(marker, str) for marker in observed_prefix
                        )
                    )
                )
            ):
                raise ValueError("invalid watcher generation observation")
            changed_generation = (
                observed.get("fingerprint") != reservation.get("fingerprint")
            )
            if changed_generation:
                active_run = self._auto_compile_run_unlocked(reservation)
                pending_run = self._auto_compile_run_unlocked(
                    reservation, pending=True
                )
                observed_key = self._auto_compile_operation_key(
                    observed_log_name, observed_fingerprint
                )
                reused_pending = (
                    pending_run is not None
                    and pending_run.operation_key == observed_key
                    and pending_run.state not in {"succeeded", "failed", "dead"}
                )
                for discarded in (
                    active_run,
                    None if reused_pending else pending_run,
                ):
                    if discarded is not None and discarded.state not in {
                        "succeeded",
                        "failed",
                        "dead",
                    }:
                        transition_run_unlocked(
                            self._connection,
                            discarded.id,
                            "failed",
                            "failed",
                            now=now_dt,
                            summary="Superseded during automatic compile takeover",
                            completed_at=now_dt,
                            redaction_env=self._redaction_env,
                        )
                reservation["status_run_id"] = (
                    pending_run.id
                    if reused_pending and pending_run is not None
                    else self._create_auto_compile_run_unlocked(
                        observed_log_name,
                        observed_fingerprint,
                        now_dt,
                        execution_token=token,
                    )
                )
            reservation.pop("pending_fingerprint", None)
            reservation.pop("pending_log_name", None)
            reservation.pop("pending_required_marker_prefix", None)
            reservation.pop("pending_status_run_id", None)
            reservation.pop("contender_token", None)
            reservation.pop("contender_predecessor_token", None)
            reservation.pop("contender_expires_at", None)
            reservation.update(observed)
            fingerprint = reservation["fingerprint"]
            reservation["token"] = token
            reservation["expires_at"] = _stored_time(owner_expiry)
            reservation["watcher_token"] = successor_token
            reservation["watcher_expires_at"] = _stored_time(watcher_expiry)
            if changed_generation:
                reservation.pop("attempt_count", None)
                reservation.pop("last_error_class", None)
            reservation.pop("next_retry_at", None)
            reservation.pop("status", None)
            self._transition_auto_compile_run_unlocked(
                reservation,
                "running",
                "generation_recovered",
                now=now_dt,
                summary="Recovered interrupted automatic compile",
            )
            self._connection.execute(
                "UPDATE queue_metadata SET value = ? WHERE key = ?",
                (
                    json.dumps(reservation, sort_keys=True, separators=(",", ":")),
                    AUTO_COMPILE_RESERVATION_KEY,
                ),
            )
            launch_successor(successor_token)
            self._connection.execute("COMMIT")
            return "claimed", fingerprint
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def record_auto_compile_failure(
        self,
        token: str,
        fingerprint: str,
        error_class: str,
        current_content: Callable[[], AutoCompileContentRead],
        *,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
        max_attempts: int = 3,
        retry_base_seconds: int = 5,
        reset_on_fingerprint_change: bool = True,
        required_marker_prefix: tuple[str, ...] | None = None,
    ) -> str:
        """Persist bounded retry state for an ordinary compile failure."""
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if max_attempts < 1 or retry_base_seconds < 1 or expires_dt <= now_dt:
            raise ValueError("invalid auto-compile retry policy")
        if required_marker_prefix is not None and not all(
            isinstance(marker, str) for marker in required_marker_prefix
        ):
            raise ValueError("invalid required marker prefix")
        safe_error = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(error_class))[:64]
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            if (
                reservation.get("token") != token
                or reservation.get("fingerprint") != fingerprint
            ):
                self._connection.execute("COMMIT")
                return "lost"
            read_status, observed, observed_markers = self._validate_auto_compile_read(
                current_content()
            )
            active_observed = read_status == "uncompiled"
            promoted_log_name = reservation.get("log_name")
            promoted_marker_prefix: tuple[str, ...] | None = None
            stored_prefix_value = reservation.get("required_marker_prefix")
            stored_prefix = (
                tuple(stored_prefix_value)
                if isinstance(stored_prefix_value, list)
                and all(isinstance(marker, str) for marker in stored_prefix_value)
                else None
            )
            marker_history_valid = True
            if observed_markers is not None and stored_prefix is not None:
                marker_history_valid = (
                    len(observed_markers) >= len(stored_prefix)
                    and observed_markers[: len(stored_prefix)] == stored_prefix
                )
                if marker_history_valid and len(observed_markers) > len(
                    stored_prefix
                ):
                    promoted_marker_prefix = observed_markers
            if not marker_history_valid:
                observed = fingerprint
            if (
                read_status == "covered"
                and marker_history_valid
                and required_marker_prefix is None
                and reservation.get("pending_log_name")
                != reservation.get("log_name")
                and isinstance(reservation.get("pending_fingerprint"), str)
                and isinstance(reservation.get("pending_log_name"), str)
            ):
                observed = reservation["pending_fingerprint"]
                promoted_log_name = reservation["pending_log_name"]
                pending_prefix = reservation.get("pending_required_marker_prefix")
                if isinstance(pending_prefix, list) and all(
                    isinstance(marker, str) for marker in pending_prefix
                ):
                    promoted_marker_prefix = tuple(pending_prefix)
                active_observed = False
            if read_status == "unreadable" or required_marker_prefix is not None:
                observed = fingerprint
            if required_marker_prefix is not None:
                reservation["required_marker_prefix"] = list(required_marker_prefix)
            elif promoted_marker_prefix is not None:
                reservation["required_marker_prefix"] = list(
                    promoted_marker_prefix
                )
            if observed is None:
                observed = fingerprint
            changed = observed != fingerprint
            if changed and reset_on_fingerprint_change:
                reservation["fingerprint"] = observed
                reservation["log_name"] = promoted_log_name
                reservation.pop("attempt_count", None)
                reservation.pop("last_error_class", None)
                preserve_other_log = (
                    active_observed
                    and reservation.get("pending_log_name")
                    != reservation.get("log_name")
                )
                if not preserve_other_log:
                    reservation.pop("pending_fingerprint", None)
                    reservation.pop("pending_log_name", None)
                    reservation.pop("pending_required_marker_prefix", None)
                reservation["status"] = "retry_wait"
                reservation["next_retry_at"] = _stored_time(now_dt)
                reservation["expires_at"] = _stored_time(now_dt)
                reservation["watcher_expires_at"] = _stored_time(expires_dt)
                outcome = "retry_wait"
            else:
                if changed:
                    reservation["fingerprint"] = observed
                    reservation["log_name"] = promoted_log_name
                    preserve_other_log = (
                        active_observed
                        and reservation.get("pending_log_name")
                        != reservation.get("log_name")
                    )
                    if not preserve_other_log:
                        reservation.pop("pending_fingerprint", None)
                        reservation.pop("pending_log_name", None)
                        reservation.pop("pending_required_marker_prefix", None)
                attempt = int(reservation.get("attempt_count", 0)) + 1
                reservation["attempt_count"] = attempt
                reservation["last_error_class"] = safe_error or "compile_failure"
                if attempt >= max_attempts:
                    reservation["status"] = "failed"
                    reservation["expires_at"] = _stored_time(now_dt)
                    reservation.pop("next_retry_at", None)
                    outcome = "failed"
                else:
                    delay = retry_base_seconds * (2 ** (attempt - 1))
                    retry_at = now_dt + timedelta(seconds=delay)
                    reservation["status"] = "retry_wait"
                    reservation["next_retry_at"] = _stored_time(retry_at)
                    reservation["expires_at"] = _stored_time(retry_at)
                    reservation["watcher_expires_at"] = _stored_time(expires_dt)
                    outcome = "retry_wait"
            if changed:
                prior_run = self._auto_compile_run_unlocked(reservation)
                if (
                    prior_run is not None
                    and prior_run.state not in {"succeeded", "failed", "dead"}
                ):
                    transition_run_unlocked(
                        self._connection,
                        prior_run.id,
                        "failed",
                        "failed",
                        now=now_dt,
                        summary="Generation changed before compile completed",
                        completed_at=now_dt,
                        redaction_env=self._redaction_env,
                    )
                reservation["status_run_id"] = self._create_auto_compile_run_unlocked(
                    str(reservation["log_name"]), str(observed), now_dt
                )
            run = self._auto_compile_run_unlocked(reservation)
            if (
                not changed
                and run is not None
                and run.state not in {"succeeded", "failed", "dead"}
            ):
                if outcome == "retry_wait":
                    if run.state == "queued":
                        self._transition_auto_compile_run_unlocked(
                            reservation,
                            "running",
                            "staging_started",
                            now=now_dt,
                        )
                    self._transition_auto_compile_run_unlocked(
                        reservation,
                        "retrying",
                        "retry_wait",
                        now=now_dt,
                        summary=safe_error or "compile_failure",
                        error=safe_error or "compile_failure",
                        level="warning",
                        message=safe_error or "compile_failure",
                        details={"retry_at": reservation["next_retry_at"]},
                    )
                else:
                    self._transition_auto_compile_run_unlocked(
                        reservation,
                        "dead",
                        "dead",
                        now=now_dt,
                        summary=safe_error or "compile_failure",
                        error=safe_error or "compile_failure",
                        level="error",
                        message=safe_error or "compile_failure",
                    )
            self._connection.execute(
                "UPDATE queue_metadata SET value = ? WHERE key = ?",
                (
                    json.dumps(reservation, sort_keys=True, separators=(",", ":")),
                    AUTO_COMPILE_RESERVATION_KEY,
                ),
            )
            self._connection.execute("COMMIT")
            return outcome
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def defer_auto_compile_generation(
        self,
        token: str,
        fingerprint: str,
        current_content: Callable[[], AutoCompileContentRead],
        *,
        compiler_lock_held: bool = False,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
    ) -> str | None:
        """Retain an exit-75 request until the child lock or marker changes."""
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if expires_dt <= now_dt:
            raise ValueError("deferred auto-compile expiry must be in the future")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            if (
                reservation.get("token") != token
                or reservation.get("fingerprint") != fingerprint
            ):
                self._connection.execute("COMMIT")
                return None
            if compiler_lock_held:
                reservation["expires_at"] = _stored_time(expires_dt)
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return fingerprint
            read_status, observed, observed_markers = self._validate_auto_compile_read(
                current_content()
            )
            required_prefix_value = reservation.get("required_marker_prefix")
            required_prefix = (
                tuple(required_prefix_value)
                if isinstance(required_prefix_value, list)
                and all(isinstance(marker, str) for marker in required_prefix_value)
                else None
            )
            marker_history_valid = False
            marker_advanced = False
            if observed_markers is not None and required_prefix is not None:
                marker_history_valid = (
                    len(observed_markers) >= len(required_prefix)
                    and observed_markers[: len(required_prefix)] == required_prefix
                )
                marker_advanced = marker_history_valid and len(
                    observed_markers
                ) > len(required_prefix)
            if (
                read_status == "unreadable"
                or not marker_history_valid
                or (read_status == "covered" and not marker_advanced)
            ):
                reservation["expires_at"] = _stored_time(expires_dt)
                reservation["status"] = "read_wait"
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return fingerprint
            active_observed = read_status == "uncompiled"
            promoted_log_name = reservation.get("log_name")
            promoted_marker_prefix = (
                observed_markers if marker_advanced else required_prefix
            )
            if (
                read_status == "covered"
                and reservation.get("pending_log_name")
                != reservation.get("log_name")
                and isinstance(reservation.get("pending_fingerprint"), str)
                and isinstance(reservation.get("pending_log_name"), str)
            ):
                observed = reservation["pending_fingerprint"]
                promoted_log_name = reservation["pending_log_name"]
                pending_prefix = reservation.get("pending_required_marker_prefix")
                if not isinstance(pending_prefix, list) or not all(
                    isinstance(marker, str) for marker in pending_prefix
                ):
                    reservation["expires_at"] = _stored_time(expires_dt)
                    reservation["status"] = "read_wait"
                    self._connection.execute(
                        "UPDATE queue_metadata SET value = ? WHERE key = ?",
                        (
                            json.dumps(
                                reservation,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            AUTO_COMPILE_RESERVATION_KEY,
                        ),
                    )
                    self._connection.execute("COMMIT")
                    return fingerprint
                promoted_marker_prefix = tuple(pending_prefix)
                active_observed = False
            if read_status == "covered" and observed is None:
                completed_run = self._auto_compile_run_unlocked(reservation)
                if completed_run is not None and completed_run.state in {
                    "queued",
                    "retrying",
                }:
                    self._transition_auto_compile_run_unlocked(
                        reservation,
                        "running",
                        "generation_recovered",
                        now=now_dt,
                        summary="Recovered completed automatic compile",
                    )
                self._transition_auto_compile_run_unlocked(
                    reservation,
                    "succeeded",
                    "succeeded",
                    now=now_dt,
                    summary=f"Compiled {reservation.get('log_name', 'daily log')}",
                )
                self._delete_auto_compile_reservation_unlocked(
                    reservation,
                    now=now_dt,
                    active_succeeded=True,
                    reason=f"Compiled {reservation.get('log_name', 'daily log')}",
                )
                self._connection.execute("COMMIT")
                return None
            changed_generation = observed != fingerprint
            if changed_generation:
                active_run = self._auto_compile_run_unlocked(reservation)
                if read_status == "covered" and marker_advanced:
                    finishing = active_run
                    if finishing is not None and finishing.state in {
                        "queued",
                        "retrying",
                    }:
                        self._transition_auto_compile_run_unlocked(
                            reservation,
                            "running",
                            "staging_started",
                            now=now_dt,
                        )
                    self._transition_auto_compile_run_unlocked(
                        reservation,
                        "succeeded",
                        "succeeded",
                        now=now_dt,
                        summary=f"Compiled {reservation.get('log_name', 'daily log')}",
                    )
                pending_run = self._auto_compile_run_unlocked(
                    reservation, pending=True
                )
                expected_key = self._auto_compile_operation_key(
                    str(promoted_log_name), str(observed)
                )
                reused_pending = (
                    pending_run is not None
                    and pending_run.operation_key == expected_key
                    and pending_run.state not in {"succeeded", "failed", "dead"}
                )
                for discarded in (
                    active_run
                    if not (read_status == "covered" and marker_advanced)
                    else None,
                    None if reused_pending else pending_run,
                ):
                    if discarded is not None and discarded.state not in {
                        "succeeded",
                        "failed",
                        "dead",
                    }:
                        transition_run_unlocked(
                            self._connection,
                            discarded.id,
                            "failed",
                            "failed",
                            now=now_dt,
                            summary="Superseded during deferred automatic compile",
                            completed_at=now_dt,
                            redaction_env=self._redaction_env,
                        )
                reservation["status_run_id"] = (
                    pending_run.id
                    if reused_pending and pending_run is not None
                    else self._create_auto_compile_run_unlocked(
                        str(promoted_log_name),
                        str(observed),
                        now_dt,
                        execution_token=str(reservation.get("token", "deferred")),
                    )
                )
                reservation.pop("pending_status_run_id", None)
            reservation["fingerprint"] = observed
            reservation["log_name"] = promoted_log_name
            if promoted_marker_prefix is not None:
                reservation["required_marker_prefix"] = list(
                    promoted_marker_prefix
                )
            reservation["expires_at"] = _stored_time(expires_dt)
            preserve_other_log = (
                active_observed
                and reservation.get("pending_log_name")
                != reservation.get("log_name")
            )
            if observed != fingerprint and not preserve_other_log:
                reservation.pop("pending_fingerprint", None)
                reservation.pop("pending_log_name", None)
                reservation.pop("pending_required_marker_prefix", None)
            self._connection.execute(
                "UPDATE queue_metadata SET value = ? WHERE key = ?",
                (
                    json.dumps(reservation, sort_keys=True, separators=(",", ":")),
                    AUTO_COMPILE_RESERVATION_KEY,
                ),
            )
            self._connection.execute("COMMIT")
            return observed
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def auto_compile_reservation(
        self,
        token: str,
        *,
        now: datetime | str | int | float,
    ) -> tuple[
        str, str | None, str | None, str | None, tuple[str, ...] | None, int
    ] | None:
        """Return an owned active/pending fingerprint and source-name pair."""
        now_dt = _datetime(now)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            if row is None:
                self._connection.execute("COMMIT")
                return None
            reservation = json.loads(row["value"])
            if self._ensure_auto_compile_links_unlocked(reservation, now_dt):
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
            unexpired = _datetime(reservation["expires_at"]) > now_dt
            self._connection.execute("COMMIT")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            return None
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        fingerprint = reservation.get("fingerprint")
        pending = reservation.get("pending_fingerprint")
        if (
            reservation.get("token") != token
            or not unexpired
            or not isinstance(fingerprint, str)
            or not self._valid_reservation_component(fingerprint)
            or (
                pending is not None
                and (
                    not isinstance(pending, str)
                    or not self._valid_reservation_component(pending)
                )
            )
        ):
            return None
        log_name = reservation.get("log_name")
        pending_log_name = reservation.get("pending_log_name")
        if log_name is not None and (
            not isinstance(log_name, str) or Path(log_name).name != log_name
        ):
            return None
        if pending_log_name is not None and (
            not isinstance(pending_log_name, str)
            or Path(pending_log_name).name != pending_log_name
        ):
            return None
        required_marker_prefix = reservation.get("required_marker_prefix")
        if required_marker_prefix is not None:
            if not isinstance(required_marker_prefix, list) or not all(
                isinstance(marker, str) for marker in required_marker_prefix
            ):
                return None
            required_markers = tuple(required_marker_prefix)
        else:
            required_markers = None
        run = self._auto_compile_run_unlocked(reservation)
        expected_key = (
            self._auto_compile_operation_key(log_name, fingerprint)
            if isinstance(log_name, str)
            else None
        )
        if run is None or not (
            run.operation_key == expected_key
            or (
                expected_key is not None
                and run.operation_key is not None
                and run.operation_key.startswith(f"{expected_key}:")
            )
        ):
            return None
        return (
            fingerprint,
            pending,
            log_name,
            pending_log_name,
            required_markers,
            run.id,
        )

    def promote_pending_auto_compile(
        self,
        token: str,
        fingerprint: str,
        pending_fingerprint: str,
        current_content: Callable[[], AutoCompileContentRead],
        *,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
    ) -> bool:
        """Promote pending work only after validating the active marker history."""
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if expires_dt <= now_dt:
            raise ValueError("auto-compile reservation expiry must be in the future")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            try:
                unexpired = _datetime(reservation["expires_at"]) > now_dt
            except (KeyError, TypeError, ValueError):
                unexpired = False
            matched = (
                reservation.get("token") == token
                and reservation.get("fingerprint") == fingerprint
                and reservation.get("pending_fingerprint") == pending_fingerprint
            )
            if not matched or not unexpired:
                self._connection.execute("COMMIT")
                return False
            if self._active_work_unlocked():
                reservation["expires_at"] = _stored_time(now_dt)
                reservation["status"] = "queue_wait"
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return False
            read_status, observed_fingerprint, observed_markers = (
                self._validate_auto_compile_read(current_content())
            )
            required_prefix_value = reservation.get("required_marker_prefix")
            required_prefix = (
                tuple(required_prefix_value)
                if isinstance(required_prefix_value, list)
                and all(isinstance(marker, str) for marker in required_prefix_value)
                else None
            )
            marker_history_valid = False
            marker_advanced = False
            if observed_markers is not None and required_prefix is not None:
                marker_history_valid = (
                    len(observed_markers) >= len(required_prefix)
                    and observed_markers[: len(required_prefix)] == required_prefix
                )
                marker_advanced = marker_history_valid and len(
                    observed_markers
                ) > len(required_prefix)
            same_log = reservation.get("pending_log_name") == reservation.get(
                "log_name"
            )
            current_generation_matches = (
                read_status == "uncompiled"
                and observed_fingerprint == pending_fingerprint
                if same_log
                else read_status == "covered" and marker_advanced
            )
            if not marker_history_valid or not current_generation_matches:
                reservation["expires_at"] = _stored_time(now_dt)
                reservation["status"] = "read_wait"
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return False
            reservation["fingerprint"] = pending_fingerprint
            reservation.pop("pending_fingerprint", None)
            active_run = self._auto_compile_run_unlocked(reservation)
            if (
                active_run is not None
                and active_run.state not in {"succeeded", "failed", "dead"}
            ):
                transition_run_unlocked(
                    self._connection,
                    active_run.id,
                    "failed",
                    "failed",
                    now=now_dt,
                    summary="Superseded before automatic compile launch",
                    completed_at=now_dt,
                    redaction_env=self._redaction_env,
                )
            pending_run = self._auto_compile_run_unlocked(reservation, pending=True)
            reservation["status_run_id"] = (
                pending_run.id
                if pending_run is not None
                else self._create_auto_compile_run_unlocked(
                    str(reservation.get("pending_log_name", reservation.get("log_name"))),
                    pending_fingerprint,
                    now_dt,
                )
            )
            reservation.pop("pending_status_run_id", None)
            pending_log_name = reservation.pop("pending_log_name", None)
            if isinstance(pending_log_name, str):
                reservation["log_name"] = pending_log_name
            pending_prefix = reservation.pop("pending_required_marker_prefix", None)
            if isinstance(pending_prefix, list) and all(
                isinstance(marker, str) for marker in pending_prefix
            ):
                reservation["required_marker_prefix"] = pending_prefix
            reservation["expires_at"] = _stored_time(expires_dt)
            self._connection.execute(
                "UPDATE queue_metadata SET value = ? WHERE key = ?",
                (
                    json.dumps(reservation, sort_keys=True, separators=(",", ":")),
                    AUTO_COMPILE_RESERVATION_KEY,
                ),
            )
            self._connection.execute("COMMIT")
            return True
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def finish_auto_compile_generation(
        self,
        token: str,
        fingerprint: str,
        current_content: Callable[[], AutoCompileContentRead],
        *,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
    ) -> str | None:
        """Release a finished generation or atomically promote later content.

        A pending fingerprint records that a later idle drain observed new
        content. The compiler's marker changes the actual fingerprint, so the
        post-compile fingerprint, rather than the stale pending value, becomes
        the next generation. The callback reads that fingerprint while the
        immediate transaction serializes the scheduler's reservation update.
        """
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if expires_dt <= now_dt:
            raise ValueError("auto-compile reservation expiry must be in the future")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            matched = (
                reservation.get("token") == token
                and reservation.get("fingerprint") == fingerprint
            )
            if not matched:
                self._connection.execute("COMMIT")
                return None
            if self._active_work_unlocked():
                reservation["expires_at"] = _stored_time(now_dt)
                reservation["status"] = "queue_wait"
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return None
            read_status, observed_fingerprint, observed_markers = (
                self._validate_auto_compile_read(current_content())
            )
            required_prefix_value = reservation.get("required_marker_prefix")
            required_prefix = (
                tuple(required_prefix_value)
                if isinstance(required_prefix_value, list)
                and all(isinstance(marker, str) for marker in required_prefix_value)
                else None
            )
            if observed_markers is not None and required_prefix is not None:
                marker_history_valid = (
                    len(observed_markers) >= len(required_prefix)
                    and observed_markers[: len(required_prefix)] == required_prefix
                )
                marker_advanced = marker_history_valid and len(
                    observed_markers
                ) > len(required_prefix)
                if not marker_history_valid or not marker_advanced:
                    reservation["expires_at"] = _stored_time(now_dt)
                    reservation["status"] = "read_wait"
                    self._connection.execute(
                        "UPDATE queue_metadata SET value = ? WHERE key = ?",
                        (
                            json.dumps(
                                reservation,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            AUTO_COMPILE_RESERVATION_KEY,
                        ),
                    )
                    self._connection.execute("COMMIT")
                    return fingerprint
            active_observed = read_status == "uncompiled"
            promoted_log_name = reservation.get("log_name")
            promoted_marker_prefix = observed_markers
            if (
                read_status == "covered"
                and reservation.get("pending_log_name")
                != reservation.get("log_name")
                and isinstance(reservation.get("pending_fingerprint"), str)
                and isinstance(reservation.get("pending_log_name"), str)
            ):
                observed_fingerprint = reservation["pending_fingerprint"]
                promoted_log_name = reservation["pending_log_name"]
                pending_prefix = reservation.get("pending_required_marker_prefix")
                if isinstance(pending_prefix, list) and all(
                    isinstance(marker, str) for marker in pending_prefix
                ):
                    promoted_marker_prefix = tuple(pending_prefix)
                active_observed = False
            if read_status == "unreadable":
                reservation["expires_at"] = _stored_time(now_dt)
                reservation["status"] = "read_wait"
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return fingerprint
            should_promote = (
                observed_fingerprint is not None
                and observed_fingerprint != fingerprint
            )
            finishing_run = self._auto_compile_run_unlocked(reservation)
            if finishing_run is not None and finishing_run.state in {
                "queued",
                "retrying",
            }:
                self._transition_auto_compile_run_unlocked(
                    reservation,
                    "running",
                    "staging_started",
                    now=now_dt,
                )
            self._transition_auto_compile_run_unlocked(
                reservation,
                "succeeded",
                "succeeded",
                now=now_dt,
                summary=f"Compiled {reservation.get('log_name', 'daily log')}",
            )
            if should_promote:
                reservation["fingerprint"] = observed_fingerprint
                reservation["log_name"] = promoted_log_name
                pending_run = self._auto_compile_run_unlocked(
                    reservation, pending=True
                )
                expected_key = self._auto_compile_operation_key(
                    str(promoted_log_name), str(observed_fingerprint)
                )
                reservation["status_run_id"] = (
                    pending_run.id
                    if pending_run is not None
                    and pending_run.operation_key == expected_key
                    else self._create_auto_compile_run_unlocked(
                        str(promoted_log_name), str(observed_fingerprint), now_dt
                    )
                )
                if promoted_marker_prefix is not None:
                    reservation["required_marker_prefix"] = list(
                        promoted_marker_prefix
                    )
                preserve_other_log = (
                    active_observed
                    and reservation.get("pending_log_name")
                    != reservation.get("log_name")
                )
                if not preserve_other_log:
                    reservation.pop("pending_fingerprint", None)
                    reservation.pop("pending_log_name", None)
                    reservation.pop("pending_required_marker_prefix", None)
                    reservation.pop("pending_status_run_id", None)
                reservation["expires_at"] = _stored_time(expires_dt)
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return observed_fingerprint
            self._delete_auto_compile_reservation_unlocked(
                reservation,
                now=now_dt,
                active_succeeded=True,
                reason=f"Compiled {reservation.get('log_name', 'daily log')}",
            )
            self._connection.execute("COMMIT")
            return None
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def release_auto_compile(self, token: str, fingerprint: str) -> bool:
        """Release a reservation only when both opaque identities match."""
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            matched = False
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
                matched = (
                    reservation.get("token") == token
                    and reservation.get("fingerprint") == fingerprint
                )
            if matched:
                self._delete_auto_compile_reservation_unlocked(
                    reservation,
                    now=self._now(),
                    active_succeeded=False,
                    reason="Automatic compile reservation released",
                )
            self._connection.execute("COMMIT")
            return matched
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def renew_auto_compile(
        self,
        token: str,
        fingerprint: str,
        *,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
    ) -> bool:
        """Extend an unexpired reservation held by the same coordinator."""
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if expires_dt <= now_dt:
            raise ValueError("auto-compile reservation expiry must be in the future")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            matched = (
                reservation.get("token") == token
                and reservation.get("fingerprint") == fingerprint
            )
            try:
                unexpired = _datetime(reservation["expires_at"]) > now_dt
            except (KeyError, TypeError, ValueError):
                unexpired = False
            if not matched or not unexpired:
                self._connection.execute("COMMIT")
                return False
            reservation["expires_at"] = _stored_time(expires_dt)
            self._connection.execute(
                "UPDATE queue_metadata SET value = ? WHERE key = ?",
                (
                    json.dumps(
                        reservation, sort_keys=True, separators=(",", ":")
                    ),
                    AUTO_COMPILE_RESERVATION_KEY,
                ),
            )
            self._connection.execute("COMMIT")
            return True
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def launch_reserved_auto_compile(
        self,
        token: str,
        fingerprint: str,
        launch: Callable[[], object],
        *,
        now: datetime | str | int | float,
    ) -> object | None:
        """Recheck reservation and queue idleness atomically with process launch.

        The immediate transaction serializes enqueues until ``launch`` returns,
        so work that commits before the child launch suppresses compilation and
        work that commits afterward belongs to a later drain.
        """
        now_dt = _datetime(now)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT value FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
            ).fetchone()
            reservation = {}
            if row is not None:
                try:
                    reservation = json.loads(row["value"])
                except (TypeError, json.JSONDecodeError):
                    reservation = {}
            matched = (
                reservation.get("token") == token
                and reservation.get("fingerprint") == fingerprint
            )
            try:
                unexpired = _datetime(reservation["expires_at"]) > now_dt
            except (KeyError, TypeError, ValueError):
                unexpired = False
            if not matched or not unexpired:
                self._connection.execute("COMMIT")
                return None
            if self._active_work_unlocked():
                reservation["expires_at"] = _stored_time(now_dt)
                reservation["status"] = "queue_wait"
                self._connection.execute(
                    "UPDATE queue_metadata SET value = ? WHERE key = ?",
                    (
                        json.dumps(
                            reservation, sort_keys=True, separators=(",", ":")
                        ),
                        AUTO_COMPILE_RESERVATION_KEY,
                    ),
                )
                self._connection.execute("COMMIT")
                return None
            run = self._auto_compile_run_unlocked(reservation)
            if run is not None and run.state in {"queued", "retrying"}:
                self._transition_auto_compile_run_unlocked(
                    reservation,
                    "running",
                    "staging_started",
                    now=now_dt,
                )
            process = launch()
            self._connection.execute("COMMIT")
            return process
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def attempts_for(self, job_id: int) -> list[ProviderAttempt]:
        rows = self._connection.execute(
            "SELECT * FROM provider_attempts WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
        return [
            ProviderAttempt(
                id=row["id"],
                job_id=row["job_id"],
                provider=row["provider"],
                model=row["model"],
                task=row["task"],
                started_at=_loaded_time(row["started_at"]),
                ended_at=_loaded_time(row["ended_at"]),
                outcome=row["outcome"],
                reason=row["reason"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                elapsed_ms=row["elapsed_ms"],
                legacy_cost_usd=row["legacy_cost_usd"],
            )
            for row in rows
        ]


# Descriptive alias for callers that prefer the storage-specific name.
SQLiteQueue = QueueRepository
JobQueue = QueueRepository
MemoryQueue = QueueRepository
