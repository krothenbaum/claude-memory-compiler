"""Durable SQLite queue for memory compiler jobs and provider attempts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sqlite3
import sysconfig
from typing import Callable, Literal

try:
    from .providers import ProviderResult
    from .transcripts import NormalizedSession, render_turns
except ImportError:  # Direct execution with scripts/ on sys.path.
    from providers import ProviderResult
    from transcripts import NormalizedSession, render_turns


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
SCHEMA_VERSION = 1
DEFAULT_MAX_ATTEMPTS = 5
MAX_ERROR_CHARS = 1_000


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
    ) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self._clock = clock
        self._connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def __enter__(self) -> "QueueRepository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _migrate(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"queue schema {version} is newer than supported version {SCHEMA_VERSION}"
            )
        if version == SCHEMA_VERSION:
            return
        try:
            self._connection.executescript(
                """
                BEGIN IMMEDIATE;
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
                );
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
                    elapsed_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_status_available_idx
                    ON jobs(status, available_at);
                CREATE INDEX IF NOT EXISTS jobs_lease_expiry_idx
                    ON jobs(lease_expires_at);
                PRAGMA user_version = 1;
                COMMIT;
                """
            )
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _now(self) -> datetime:
        return _datetime(self._clock())

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
        now = _stored_time(self._now())
        payload = json.dumps(
            {
                "timestamp": session.timestamp,
                "turns": [asdict(turn) for turn in session.turns],
                "rendered_context": render_turns(session),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
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
            WHERE kind = 'capture' AND source_agent = ? AND session_id = ? AND source_hash = ?
            """,
            (session.agent, session.session_id, session.source_hash),
        ).fetchone()
        if row is None:  # Defensive: the insert/select are on one connection.
            raise RuntimeError("enqueued capture could not be read back")
        return EnqueueResult(self._job(row), created)

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

    def record_attempt(self, job_id: int, result: ProviderResult) -> None:
        ended = self._now()
        started = ended - timedelta(milliseconds=max(0, result.elapsed_ms))
        self._connection.execute(
            """
            INSERT INTO provider_attempts (
                job_id, provider, model, task, started_at, ended_at, outcome,
                reason, input_tokens, output_tokens, elapsed_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                result.provider,
                result.model,
                result.task.value,
                _stored_time(started),
                _stored_time(ended),
                result.outcome,
                result.reason[:MAX_ERROR_CHARS] if result.reason else None,
                result.input_tokens,
                result.output_tokens,
                max(0, result.elapsed_ms),
            ),
        )

    def complete(self, job_id: int, owner: str) -> None:
        now = _stored_time(self._now())
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
            now = _stored_time(self._now())
            dead = row["attempt_count"] >= self.max_attempts
            cursor = self._connection.execute(
                """
                UPDATE jobs
                SET status = ?, available_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL, last_error = ?, updated_at = ?, completed_at = ?
                WHERE id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (
                    "dead" if dead else "failed",
                    _stored_time(available_at),
                    " ".join(error.split())[:MAX_ERROR_CHARS],
                    now,
                    now if dead else None,
                    job_id,
                    owner,
                ),
            )
            if cursor.rowcount != 1:
                raise LeaseOwnershipError(f"job {job_id} is not leased by {owner}")
            self._connection.execute("COMMIT")
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def recover_stale(self, now: datetime | str | int | float) -> int:
        now_value = _stored_time(now)
        cursor = self._connection.execute(
            """
            UPDATE jobs
            SET status = CASE WHEN attempt_count >= ? THEN 'dead' ELSE 'failed' END,
                available_at = ?, lease_owner = NULL, lease_expires_at = NULL,
                last_error = 'worker lease expired', updated_at = ?,
                completed_at = CASE WHEN attempt_count >= ? THEN ? ELSE NULL END
            WHERE status = 'leased' AND lease_expires_at <= ?
            """,
            (self.max_attempts, now_value, now_value, self.max_attempts, now_value, now_value),
        )
        return cursor.rowcount

    def get_job(self, job_id: int) -> Job:
        row = self._connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job(row)

    def count_jobs(self) -> int:
        return self._connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

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
            )
            for row in rows
        ]


# Descriptive alias for callers that prefer the storage-specific name.
SQLiteQueue = QueueRepository
JobQueue = QueueRepository
MemoryQueue = QueueRepository
