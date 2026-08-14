"""Durable SQLite queue for memory compiler jobs and provider attempts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sysconfig
from typing import Callable, Literal, Mapping

try:
    from .providers import ProviderResult
    from .transcripts import NormalizedSession, render_turns
    from .usage import (
        UnsafeUsagePathError,
        UsageRecord,
        append_usage_record,
        logged_provider_attempt_ids,
        recover_usage_log,
    )
except ImportError:  # Direct execution with scripts/ on sys.path.
    from providers import ProviderResult
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
SCHEMA_VERSION = 2
DEFAULT_MAX_ATTEMPTS = 5
MAX_ERROR_CHARS = 1_000
AUTO_COMPILE_RESERVATION_KEY = "auto_compile_reservation"
_SECRET_NAMES = {"ANTHROPIC_API_KEY", "CLAUDE_API_KEY"}
_SECRET_SUFFIXES = ("_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD")


class LeaseOwnershipError(RuntimeError):
    """Raised when a worker mutates a lease it does not own."""


def normalize_persistence_reason(
    reason: object,
    env: Mapping[str, str],
) -> str:
    """Bound and redact metadata persisted at queue failure boundaries."""
    normalized = " ".join(str(reason).split()) or "unspecified failure"
    secrets = {
        value
        for name, value in env.items()
        if value
        and (
            name in _SECRET_NAMES
            or name.startswith("OPENAI_")
            or name.startswith("AZURE_OPENAI_")
            or name.endswith(_SECRET_SUFFIXES)
        )
    }
    for secret in sorted(secrets, key=len, reverse=True):
        normalized = normalized.replace(secret, "[REDACTED]")
    return normalized[:MAX_ERROR_CHARS]


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

    def _migrate(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"queue schema {version} is newer than supported version {SCHEMA_VERSION}"
            )
        if version == SCHEMA_VERSION:
            return
        if version == 1:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "ALTER TABLE provider_attempts ADD COLUMN legacy_cost_usd REAL"
                )
                self._connection.execute(
                    "CREATE TABLE queue_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self._connection.execute(
                    "INSERT INTO queue_metadata(key, value) VALUES ('queue_id', lower(hex(randomblob(16))))"
                )
                self._connection.execute("PRAGMA user_version = 2")
                self._connection.execute("COMMIT")
                return
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
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
                    elapsed_ms INTEGER NOT NULL,
                    legacy_cost_usd REAL
                );
                CREATE INDEX IF NOT EXISTS jobs_status_available_idx
                    ON jobs(status, available_at);
                CREATE INDEX IF NOT EXISTS jobs_lease_expiry_idx
                    ON jobs(lease_expires_at);
                CREATE TABLE IF NOT EXISTS queue_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO queue_metadata(key, value)
                    VALUES ('queue_id', lower(hex(randomblob(16))));
                PRAGMA user_version = 2;
                COMMIT;
                """
            )
        except BaseException:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _now(self) -> datetime:
        return _datetime(self._clock())

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
                    normalize_persistence_reason(error, self._redaction_env),
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

    def _active_work_unlocked(self) -> bool:
        return self._connection.execute(
            """
            SELECT 1 FROM jobs
            WHERE status IN ('pending', 'failed', 'leased')
            LIMIT 1
            """
        ).fetchone() is not None

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
                    "expires_at": _stored_time(expires_dt),
                }
                role = "owner"
            else:
                reservation["pending_fingerprint"] = fingerprint
                reservation["pending_log_name"] = log_name
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
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
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
            try:
                owner_live = _datetime(reservation["expires_at"]) > now_dt
            except (KeyError, TypeError, ValueError):
                owner_live = False
            status = reservation.get("status")
            if not reservation or status == "failed":
                reservation = {
                    "token": owner_token,
                    "fingerprint": fingerprint,
                    "log_name": log_name,
                    "expires_at": _stored_time(expires_dt),
                    "watcher_token": watchdog_token,
                    "watcher_expires_at": _stored_time(expires_dt),
                }
                roles = ("owner", "watchdog")
            elif not owner_live:
                reservation["pending_fingerprint"] = fingerprint
                reservation["pending_log_name"] = log_name
                try:
                    watcher_live = (
                        _datetime(reservation["watcher_expires_at"]) > now_dt
                        and isinstance(reservation.get("watcher_token"), str)
                    )
                except (KeyError, TypeError, ValueError):
                    watcher_live = False
                if watcher_live:
                    roles = ()
                else:
                    reservation["watcher_token"] = watchdog_token
                    reservation["watcher_expires_at"] = _stored_time(expires_dt)
                    roles = ("watchdog",)
            else:
                reservation["pending_fingerprint"] = fingerprint
                reservation["pending_log_name"] = log_name
                try:
                    watcher_live = (
                        _datetime(reservation["watcher_expires_at"]) > now_dt
                        and isinstance(reservation.get("watcher_token"), str)
                    )
                except (KeyError, TypeError, ValueError):
                    watcher_live = False
                if watcher_live:
                    roles = ()
                else:
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
                self._connection.execute(
                    "DELETE FROM queue_metadata WHERE key = ?",
                    (AUTO_COMPILE_RESERVATION_KEY,),
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
                self._connection.execute(
                    "DELETE FROM queue_metadata WHERE key = ?",
                    (AUTO_COMPILE_RESERVATION_KEY,),
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
        observe: Callable[[Mapping[str, object]], Mapping[str, str] | None],
        launch_successor: Callable[[str], None],
        *,
        now: datetime | str | int | float,
        watcher_expires_at: datetime | str | int | float,
        owner_expires_at: datetime | str | int | float,
    ) -> tuple[str, str | None]:
        """Heartbeat a watcher or atomically promote it after owner expiry."""
        if not self._valid_reservation_component(successor_token):
            raise ValueError("successor token must be a lowercase SHA-256 value")
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
            if reservation.get("watcher_token") != token:
                self._connection.execute("COMMIT")
                return "done", None
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
            observed = observe(reservation)
            if observed is None:
                self._connection.execute(
                    "DELETE FROM queue_metadata WHERE key = ?",
                    (AUTO_COMPILE_RESERVATION_KEY,),
                )
                self._connection.execute("COMMIT")
                return "done", None
            if current_owner_expiry > now_dt:
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
            reservation.pop("pending_fingerprint", None)
            reservation.pop("pending_log_name", None)
            changed_generation = (
                observed.get("fingerprint") != reservation.get("fingerprint")
            )
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
        current_fingerprint: Callable[[], str | None],
        *,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
        max_attempts: int = 3,
        retry_base_seconds: int = 5,
    ) -> str:
        """Persist bounded retry state for an ordinary compile failure."""
        now_dt = _datetime(now)
        expires_dt = _datetime(expires_at)
        if max_attempts < 1 or retry_base_seconds < 1 or expires_dt <= now_dt:
            raise ValueError("invalid auto-compile retry policy")
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
            observed = current_fingerprint()
            active_observed = observed is not None
            promoted_log_name = reservation.get("log_name")
            if (
                observed is None
                and reservation.get("pending_log_name")
                != reservation.get("log_name")
                and isinstance(reservation.get("pending_fingerprint"), str)
                and isinstance(reservation.get("pending_log_name"), str)
            ):
                observed = reservation["pending_fingerprint"]
                promoted_log_name = reservation["pending_log_name"]
            if observed is None:
                self._connection.execute(
                    "DELETE FROM queue_metadata WHERE key = ?",
                    (AUTO_COMPILE_RESERVATION_KEY,),
                )
                self._connection.execute("COMMIT")
                return "covered"
            changed = observed != fingerprint
            if changed:
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
                reservation["status"] = "retry_wait"
                reservation["next_retry_at"] = _stored_time(now_dt)
                reservation["expires_at"] = _stored_time(now_dt)
                reservation["watcher_expires_at"] = _stored_time(expires_dt)
                outcome = "retry_wait"
            else:
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
        current_fingerprint: Callable[[], str | None],
        *,
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
            observed = current_fingerprint()
            active_observed = observed is not None
            promoted_log_name = reservation.get("log_name")
            if (
                observed is None
                and reservation.get("pending_log_name")
                != reservation.get("log_name")
                and isinstance(reservation.get("pending_fingerprint"), str)
                and isinstance(reservation.get("pending_log_name"), str)
            ):
                observed = reservation["pending_fingerprint"]
                promoted_log_name = reservation["pending_log_name"]
            if observed is None:
                self._connection.execute(
                    "DELETE FROM queue_metadata WHERE key = ?",
                    (AUTO_COMPILE_RESERVATION_KEY,),
                )
                self._connection.execute("COMMIT")
                return None
            reservation["fingerprint"] = observed
            reservation["log_name"] = promoted_log_name
            reservation["expires_at"] = _stored_time(expires_dt)
            preserve_other_log = (
                active_observed
                and reservation.get("pending_log_name")
                != reservation.get("log_name")
            )
            if observed != fingerprint and not preserve_other_log:
                reservation.pop("pending_fingerprint", None)
                reservation.pop("pending_log_name", None)
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
    ) -> tuple[str, str | None] | None:
        """Return an owned, unexpired active and pending fingerprint pair."""
        now_dt = _datetime(now)
        row = self._connection.execute(
            "SELECT value FROM queue_metadata WHERE key = ?",
            (AUTO_COMPILE_RESERVATION_KEY,),
        ).fetchone()
        if row is None:
            return None
        try:
            reservation = json.loads(row["value"])
            unexpired = _datetime(reservation["expires_at"]) > now_dt
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
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
        return fingerprint, pending

    def promote_pending_auto_compile(
        self,
        token: str,
        fingerprint: str,
        pending_fingerprint: str,
        *,
        now: datetime | str | int | float,
        expires_at: datetime | str | int | float,
    ) -> bool:
        """Promote the exact pending generation while the queue remains idle."""
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
            reservation["fingerprint"] = pending_fingerprint
            reservation.pop("pending_fingerprint", None)
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
        current_fingerprint: Callable[[], str | None],
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
            observed_fingerprint = current_fingerprint()
            active_observed = observed_fingerprint is not None
            promoted_log_name = reservation.get("log_name")
            if (
                observed_fingerprint is None
                and reservation.get("pending_log_name")
                != reservation.get("log_name")
                and isinstance(reservation.get("pending_fingerprint"), str)
                and isinstance(reservation.get("pending_log_name"), str)
            ):
                observed_fingerprint = reservation["pending_fingerprint"]
                promoted_log_name = reservation["pending_log_name"]
            should_promote = (
                observed_fingerprint is not None
                and observed_fingerprint != fingerprint
            )
            if should_promote:
                reservation["fingerprint"] = observed_fingerprint
                reservation["log_name"] = promoted_log_name
                preserve_other_log = (
                    active_observed
                    and reservation.get("pending_log_name")
                    != reservation.get("log_name")
                )
                if not preserve_other_log:
                    reservation.pop("pending_fingerprint", None)
                    reservation.pop("pending_log_name", None)
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
            self._connection.execute(
                "DELETE FROM queue_metadata WHERE key = ?",
                (AUTO_COMPILE_RESERVATION_KEY,),
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
                self._connection.execute(
                    "DELETE FROM queue_metadata WHERE key = ?",
                    (AUTO_COMPILE_RESERVATION_KEY,),
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
