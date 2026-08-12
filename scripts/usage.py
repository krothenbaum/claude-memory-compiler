"""Provider-neutral usage and outcome observability.

SQLite ``provider_attempts`` remains authoritative for queued work.  This
module writes a compact, privacy-bounded JSONL projection for operators and
also records provider attempts made by foreground commands that have no job.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile
import hashlib
from typing import Literal, Mapping

try:
    from .providers import ProviderResult, RoutedResult
    from .utils import ExclusiveFileLock, _fsync_directory
except ImportError:  # Direct execution with scripts/ on sys.path.
    from providers import ProviderResult, RoutedResult
    from utils import ExclusiveFileLock, _fsync_directory


MAX_ERROR_CHARS = 1_000
MAX_USAGE_BYTES = 8 * 1024 * 1024
MAX_USAGE_LINE_BYTES = 64 * 1024
_SECRET_NAMES = {"ANTHROPIC_API_KEY", "CLAUDE_API_KEY"}
_SECRET_SUFFIXES = ("_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD")
_ATTEMPT_INDEX: dict[Path, tuple[tuple[int, int, int, int, int], set[tuple[str, int]]]] = {}


class UnsafeUsagePathError(ValueError):
    """The usage projection path escapes through an unsafe filesystem object."""


def _link_or_reparse(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse and attributes & reparse)


def _validate_usage_directory(path: Path, label: str) -> None:
    info = path.lstat()
    if _link_or_reparse(info):
        raise UnsafeUsagePathError(f"usage {label} must not be a symlink or reparse point")
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafeUsagePathError(f"usage {label} must be a directory")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise UnsafeUsagePathError(f"usage {label} has an unsafe owner")


def _prepare_usage_tree(memory_home: Path, *, create_logs: bool) -> Path | None:
    if not memory_home.exists() and not memory_home.is_symlink():
        if not create_logs:
            return None
        memory_home.mkdir(parents=True, mode=0o700, exist_ok=True)
    _validate_usage_directory(memory_home, "memory root")
    scripts = memory_home / "scripts"
    if not scripts.exists() and not scripts.is_symlink():
        if not create_logs:
            return None
        scripts.mkdir(mode=0o700, exist_ok=True)
    _validate_usage_directory(scripts, "scripts directory")
    logs = scripts / "logs"
    if not logs.exists() and not logs.is_symlink():
        if not create_logs:
            return None
        logs.mkdir(mode=0o700, exist_ok=True)
    _validate_usage_directory(logs, "logs directory")
    logs.chmod(0o700)
    return logs


def routed_invalid_output(result: RoutedResult, reason: object) -> RoutedResult:
    """Replace a routed success whose staged output failed host validation."""
    failed = ProviderResult(
        provider=result.provider,
        model=result.model,
        task=result.task,
        outcome="invalid_output",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        elapsed_ms=result.elapsed_ms,
        reason=str(reason),
    )
    attempts = result.attempts
    if attempts and attempts[-1].provider == result.provider:
        attempts = (*attempts[:-1], failed)
    else:
        attempts = (*attempts, failed)
    return RoutedResult.from_result(failed, attempts, result.fallback_reason)


def _timestamp(value: datetime | str | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _safe_text(value: object, env: Mapping[str, str]) -> str:
    text = " ".join(str(value).split())
    secrets = {
        secret
        for name, secret in env.items()
        if secret
        and (
            name in _SECRET_NAMES
            or name.startswith("OPENAI_")
            or name.startswith("AZURE_OPENAI_")
            or name.endswith(_SECRET_SUFFIXES)
        )
    }
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text[:MAX_ERROR_CHARS]


@dataclass(frozen=True)
class UsageRecord:
    provider: Literal["codex", "claude"]
    model: str
    task: str
    source_agent: Literal["claude", "codex", "system"]
    outcome: str
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: int
    timestamp: str
    job_id: int | None = None
    fallback_reason: str | None = None
    reason: str | None = None
    legacy_cost_usd: float | None = None
    provider_attempt_id: int | None = None
    queue_id: str | None = None

    @classmethod
    def from_attempt(
        cls,
        attempt: ProviderResult,
        *,
        source_agent: Literal["claude", "codex", "system"],
        timestamp: datetime | str | None = None,
        job_id: int | None = None,
        fallback_reason: str | None = None,
        legacy_cost_usd: float | None = None,
        provider_attempt_id: int | None = None,
    ) -> "UsageRecord":
        return cls(
            provider=attempt.provider,
            model=attempt.model,
            task=attempt.task.value,
            source_agent=source_agent,
            outcome=attempt.outcome,
            input_tokens=attempt.input_tokens,
            output_tokens=attempt.output_tokens,
            elapsed_ms=max(0, attempt.elapsed_ms),
            timestamp=_timestamp(timestamp),
            job_id=job_id,
            fallback_reason=fallback_reason,
            reason=attempt.reason,
            legacy_cost_usd=(
                legacy_cost_usd if attempt.provider == "claude" else None
            ),
            provider_attempt_id=provider_attempt_id,
        )

    def to_dict(self, *, env: Mapping[str, str] | None = None) -> dict[str, object]:
        source_env = os.environ if env is None else env
        value: dict[str, object] = {
            "job_id": self.job_id,
            "task": self.task,
            "source_agent": self.source_agent,
            "provider": self.provider,
            "model": self.model,
            "outcome": self.outcome,
            "fallback_reason": (
                _safe_text(self.fallback_reason, source_env)
                if self.fallback_reason is not None
                else None
            ),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp,
        }
        if self.reason is not None:
            value["reason"] = _safe_text(self.reason, source_env)
        if self.legacy_cost_usd is not None and self.provider == "claude":
            value["cost_usd"] = self.legacy_cost_usd
        if self.provider_attempt_id is not None:
            value["provider_attempt_id"] = self.provider_attempt_id
        if self.queue_id is not None:
            value["queue_id"] = self.queue_id
        return value


def _read_private_log(path: Path) -> bytes:
    if not path.exists() and not path.is_symlink():
        return b""
    info = path.lstat()
    if _link_or_reparse(info):
        raise UnsafeUsagePathError("usage log must not be a symlink or reparse point")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("usage log must be a regular file")
    if info.st_nlink != 1:
        raise ValueError("usage log must not be hard-linked")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError("usage log has an unsafe owner")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_USAGE_BYTES + 1)
            if len(data) > MAX_USAGE_BYTES:
                raise ValueError("usage log exceeds byte limit")
            return data
    finally:
        os.close(descriptor)


def _attempt_key(value: Mapping[str, object]) -> tuple[str, int] | None:
    attempt_id = value.get("provider_attempt_id")
    queue_id = value.get("queue_id")
    if (
        isinstance(queue_id, str)
        and queue_id
        and isinstance(attempt_id, int)
        and not isinstance(attempt_id, bool)
    ):
        return queue_id, attempt_id
    return None


def _contains_attempt(data: bytes, record: UsageRecord) -> bool:
    expected = (
        (record.queue_id, record.provider_attempt_id)
        if record.queue_id and record.provider_attempt_id is not None
        else None
    )
    if expected is None:
        return False
    for line in data.splitlines():
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("usage log contains malformed JSONL") from exc
        if not isinstance(value, dict):
            raise ValueError("usage log records must be JSON objects")
        if _attempt_key(value) == expected:
            return True
    return False


def _valid_usage_records(data: bytes) -> tuple[list[dict[str, object]], bytes]:
    records: list[dict[str, object]] = []
    corrupt = bytearray()
    seen: set[tuple[str, int]] = set()
    for line in data.splitlines(keepends=True):
        raw = line.rstrip(b"\r\n")
        try:
            if len(raw) > MAX_USAGE_LINE_BYTES:
                raise ValueError("usage record exceeds byte limit")
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("usage record must be an object")
        except (UnicodeError, json.JSONDecodeError, ValueError):
            corrupt.extend(line)
            continue
        key = _attempt_key(value)
        if key is not None and key in seen:
            corrupt.extend(line)
            continue
        if key is not None:
            seen.add(key)
        records.append(value)
    return records, bytes(corrupt)


def logged_provider_attempt_ids(memory_home: Path | str) -> set[tuple[str, int]]:
    """Read stable queue-attempt identities without taking the writer lock."""
    home = Path(os.path.abspath(Path(memory_home).expanduser()))
    logs = _prepare_usage_tree(home, create_logs=False)
    if logs is None:
        return set()
    return _indexed_attempts(logs)


def _usage_log_paths(logs: Path) -> tuple[Path, ...]:
    archives = tuple(sorted(logs.glob("usage.archive-*.jsonl")))
    active = logs / "usage.jsonl"
    return (*archives, active) if active.exists() or active.is_symlink() else archives


def _validate_usage_file(path: Path) -> os.stat_result:
    info = path.lstat()
    if _link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise UnsafeUsagePathError(
            "usage log must be a regular non-symlink, non-reparse file"
        )
    if info.st_nlink != 1:
        raise UnsafeUsagePathError("usage log must not be hard-linked")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise UnsafeUsagePathError("usage log has an unsafe owner")
    return info


def _log_signature(logs: Path) -> tuple[int, int, int, int, int]:
    directory = logs.stat()
    active = logs / "usage.jsonl"
    if not active.exists() and not active.is_symlink():
        return directory.st_ino, directory.st_mtime_ns, 0, 0, 0
    info = _validate_usage_file(active)
    return (
        directory.st_ino,
        directory.st_mtime_ns,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    )


def _indexed_attempts(logs: Path, *, recover_active: bool = False) -> set[tuple[str, int]]:
    signature = _log_signature(logs)
    cached = _ATTEMPT_INDEX.get(logs)
    if cached is not None and cached[0] == signature:
        return set(cached[1])
    attempts: set[tuple[str, int]] = set()
    for path in _usage_log_paths(logs):
        _validate_usage_file(path)
        records, corrupt = _valid_usage_records(_read_private_log(path))
        if corrupt:
            if path.name != "usage.jsonl":
                raise ValueError("usage archive contains malformed JSONL")
            if recover_active:
                _recover_usage_unlocked(path, _read_private_log(path))
                _ATTEMPT_INDEX.pop(logs, None)
                return _indexed_attempts(logs)
            continue
        attempts.update(
            key for value in records if (key := _attempt_key(value)) is not None
        )
    _ATTEMPT_INDEX[logs] = (signature, attempts)
    return set(attempts)


def _write_private(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_existing_quarantine(path: Path, expected: bytes) -> None:
    if not path.exists() and not path.is_symlink():
        return
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("usage quarantine must be a regular file")
    if info.st_nlink != 1:
        raise ValueError("usage quarantine must not be hard-linked")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ValueError("usage quarantine has an unsafe owner")
    if _read_private_log(path) != expected:
        raise ValueError("usage quarantine content does not match its digest")


def _recover_usage_unlocked(path: Path, original: bytes) -> bytes:
    records, corrupt = _valid_usage_records(original)
    canonical = b"".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for value in records
    )
    if not corrupt and canonical == original:
        return original
    if corrupt:
        digest = hashlib.sha256(corrupt).hexdigest()
        quarantine = path.parent / f"usage.corrupt-{digest}.jsonl"
        _validate_existing_quarantine(quarantine, corrupt)
        if not quarantine.exists():
            _write_private(quarantine, corrupt)
    _write_private(path, canonical)
    return canonical


def _validate_archive(path: Path, expected: bytes) -> None:
    if not path.exists() and not path.is_symlink():
        return
    info = path.lstat()
    if _link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        raise UnsafeUsagePathError("usage archive must be a regular non-reparse file")
    if info.st_nlink != 1:
        raise UnsafeUsagePathError("usage archive must not be hard-linked")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise UnsafeUsagePathError("usage archive has an unsafe owner")
    if _read_private_log(path) != expected:
        raise ValueError("usage archive content does not match its digest")


def _rotate_usage_log(path: Path, original: bytes) -> None:
    if not original:
        return
    digest = hashlib.sha256(original).hexdigest()
    archive = path.parent / f"usage.archive-{digest}.jsonl"
    _validate_archive(archive, original)
    if not archive.exists():
        os.replace(path, archive)
        archive.chmod(0o600)
        _fsync_directory(path.parent)
    else:
        _write_private(path, b"")


def _append_private(path: Path, serialized: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if _link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
            raise UnsafeUsagePathError("usage log must be a regular non-reparse file")
        if info.st_nlink != 1:
            raise UnsafeUsagePathError("usage log must not be hard-linked")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise UnsafeUsagePathError("usage log has an unsafe owner")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        os.write(descriptor, serialized)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_usage_record_unlocked(
    memory_home: Path,
    record: UsageRecord,
    *,
    env: Mapping[str, str],
) -> Path:
    logs = _prepare_usage_tree(memory_home, create_logs=True)
    assert logs is not None
    path = logs / "usage.jsonl"
    attempts = _indexed_attempts(logs, recover_active=True)
    expected = (
        (record.queue_id, record.provider_attempt_id)
        if record.queue_id and record.provider_attempt_id is not None
        else None
    )
    if expected is not None and expected in attempts:
        return path
    serialized = json.dumps(
        record.to_dict(env=env), sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    current_size = path.stat().st_size if path.exists() else 0
    if current_size and current_size + len(serialized) > MAX_USAGE_BYTES:
        original = _read_private_log(path)
        _rotate_usage_log(path, original)
    _append_private(path, serialized)
    if expected is not None:
        attempts.add(expected)
    _ATTEMPT_INDEX[logs] = (_log_signature(logs), attempts)
    return path


def append_usage_record(
    memory_home: Path | str,
    record: UsageRecord,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Append one compact record under the shared writer lock."""
    home = Path(os.path.abspath(Path(memory_home).expanduser()))
    source_env = dict(os.environ if env is None else env)
    _prepare_usage_tree(home, create_logs=True)
    with ExclusiveFileLock(home / "scripts/memory-writer.lock"):
        _prepare_usage_tree(home, create_logs=True)
        return _append_usage_record_unlocked(home, record, env=source_env)


def recover_usage_log(memory_home: Path | str) -> Path:
    """Quarantine malformed records and retain every recoverable valid record."""
    home = Path(os.path.abspath(Path(memory_home).expanduser()))
    logs = _prepare_usage_tree(home, create_logs=False)
    path = home / "scripts/logs/usage.jsonl"
    if logs is None:
        return path
    if not path.exists() and not path.is_symlink():
        return path
    original = _read_private_log(path)
    records, corrupt = _valid_usage_records(original)
    canonical = b"".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for value in records
    )
    if not corrupt and canonical == original:
        return path
    with ExclusiveFileLock(home / "scripts/memory-writer.lock"):
        _prepare_usage_tree(home, create_logs=False)
        original = _read_private_log(path)
        _recover_usage_unlocked(path, original)
        _ATTEMPT_INDEX.pop(logs, None)
    return path


def record_routed_usage(
    memory_home: Path | str,
    routed: RoutedResult,
    *,
    source_agent: Literal["claude", "codex", "system"],
    timestamp: datetime | str | None = None,
    job_id: int | None = None,
) -> None:
    """Append every attempt in one routed foreground operation."""
    for attempt in routed.attempts:
        append_usage_record(
            memory_home,
            UsageRecord.from_attempt(
                attempt,
                job_id=job_id,
                source_agent=source_agent,
                timestamp=timestamp,
                fallback_reason=(
                    routed.fallback_reason if attempt.provider == "claude" else None
                ),
            ),
        )
