"""Pure, bounded projection of recent structured hook failures."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

try:
    from .config import load_config
    from .privacy import normalize_persistence_reason
    from .status_store import HealthAlert
    from .utils import inspect_secure_read_file
except ImportError:  # Direct execution with scripts/ on sys.path.
    from config import load_config
    from privacy import normalize_persistence_reason
    from status_store import HealthAlert
    from utils import inspect_secure_read_file

try:
    from .queue import QueueRepository
except ImportError:  # Direct execution with scripts/ on sys.path.
    from queue import QueueRepository  # type: ignore[attr-defined]


_HOOK_COMPONENTS: Final[frozenset[str]] = frozenset(
    {"session-end", "pre-compact", "codex-session-end"}
)
_HOOK_ERROR_EVENTS: Final[frozenset[str]] = frozenset(
    {
        "malformed_input",
        "transcript_missing",
        "transcript_unreadable",
        "capture_failed",
        "queue_unavailable",
    }
)
_ALERT_WINDOW = timedelta(days=1)
_FUTURE_SKEW = timedelta(minutes=5)
_MAX_TAIL_BYTES = 1_000_000
_MAX_ALERTS = 100
_DIAGNOSTIC_EVENTS = frozenset(
    {
        "malformed_input",
        "transcript_missing",
        "transcript_unreadable",
        "capture_failed",
    }
)
_MIN_DIAGNOSTIC_SECONDS = 0.1
_MAX_DIAGNOSTIC_IDENTITY_CHARS = 256
_DIAGNOSTIC_LOCK = threading.Lock()


def _canonical_redaction_env() -> dict[str, str]:
    return {
        name: canonical
        for name, value in os.environ.items()
        if (canonical := " ".join(value.split()))
    }


def _safe_diagnostic_identity(
    value: object,
    redaction_env: dict[str, str],
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_DIAGNOSTIC_IDENTITY_CHARS
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or normalize_persistence_reason(value, redaction_env) != value
    ):
        return "unknown"
    return value


def record_hook_diagnostic(
    memory_home: Path,
    *,
    event: str,
    source_agent: object,
    session_id: object,
    project: object,
    message: object,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    token_factory: Callable[[], object] = secrets.token_hex,
) -> bool:
    """Best-effort persistence for one pre-queue hook failure occurrence."""
    if event not in _DIAGNOSTIC_EVENTS:
        return False
    if deadline is not None:
        remaining = deadline - clock()
        if remaining < _MIN_DIAGNOSTIC_SECONDS:
            return False
        busy_timeout_ms = max(1, min(100, int((remaining - 0.05) * 1_000)))
    else:
        busy_timeout_ms = 100
    redaction_env = _canonical_redaction_env()
    safe_source = (
        source_agent
        if isinstance(source_agent, str) and source_agent in {"claude", "codex"}
        else "unknown"
    )
    safe_session = _safe_diagnostic_identity(session_id, redaction_env)
    safe_project = _safe_diagnostic_identity(project, redaction_env)
    try:
        token = token_factory()
        occurrence = hashlib.sha256(str(token).encode("utf-8")).hexdigest()[:32]
        operation_key = f"hook-diagnostic:{event}:{occurrence}"
        root = Path(os.path.abspath(memory_home.expanduser()))
        config = load_config(
            {
                **os.environ,
                "AI_MEMORY_HOME": str(root),
                "CLAUDE_MEMORY_HOME": str(root),
            }
        )
        lock_timeout = max(0.0, deadline - clock()) if deadline is not None else 1.0
        if not _DIAGNOSTIC_LOCK.acquire(timeout=lock_timeout):
            return False
        try:
            with QueueRepository(
                config.queue_path,
                busy_timeout_ms=busy_timeout_ms,
                memory_home=config.root_dir,
                sync_usage=False,
                redaction_env=redaction_env,
            ) as repository:
                if deadline is not None and clock() >= deadline:
                    return False
                repository.create_failed_operation_run(
                    operation_key,
                    kind="capture",
                    source_agent=safe_source,
                    session_id=safe_session,
                    project=safe_project,
                    summary=f"Hook failure: {event}",
                    error=message if isinstance(message, str) else "hook failure",
                )
        finally:
            _DIAGNOSTIC_LOCK.release()
        return True
    except Exception:
        return False


def _safe_message(
    value: object,
    redaction_env: dict[str, str],
) -> str | None:
    if not isinstance(value, str):
        return None
    text = "".join(
        character if ord(character) >= 32 and ord(character) != 127 else " "
        for character in value
    )
    normalized = normalize_persistence_reason(text, redaction_env)
    return normalized if normalized else None


def _loaded_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)
    except (OverflowError, ValueError):
        return None


def _bounded_log_tail(path: Path, *, max_bytes: int) -> bytes | None:
    try:
        identity = inspect_secure_read_file(path)
    except (FileNotFoundError, OSError, ValueError):
        return None
    descriptor = -1
    try:
        descriptor = os.open(
            identity.path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (
                hasattr(os, "getuid")
                and opened.st_uid != os.getuid()
            )
            or (
                os.name != "nt"
                and stat.S_IMODE(opened.st_mode) != 0o600
            )
            or (opened.st_dev, opened.st_ino) != (identity.device, identity.inode)
        ):
            return None
        try:
            opened_identity = inspect_secure_read_file(identity.path)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if (opened_identity.device, opened_identity.inode) != (
            identity.device,
            identity.inode,
        ):
            return None
        start = max(0, opened.st_size - max_bytes)
        preceding = b"\n"
        if start:
            os.lseek(descriptor, start - 1, os.SEEK_SET)
            preceding = os.read(descriptor, 1)
        os.lseek(descriptor, start, os.SEEK_SET)
        data = os.read(descriptor, max_bytes)
        try:
            observed = inspect_secure_read_file(identity.path)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if (observed.device, observed.inode) != (identity.device, identity.inode):
            return None
        if data and not data.endswith(b"\n"):
            final_boundary = data.rfind(b"\n")
            data = b"" if final_boundary < 0 else data[: final_boundary + 1]
        if start and preceding != b"\n":
            boundary = data.find(b"\n")
            data = b"" if boundary < 0 else data[boundary + 1 :]
        return data
    except OSError:
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def read_recent_hook_alerts(
    memory_home: Path,
    *,
    now: datetime,
    max_bytes: int = 256_000,
) -> tuple[HealthAlert, ...]:
    """Return recent recognized error records from the bounded JSONL tail."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("hook alert time must be timezone-aware")
    if max_bytes <= 0 or max_bytes > _MAX_TAIL_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {_MAX_TAIL_BYTES}")
    path = (
        Path(os.path.abspath(memory_home.expanduser()))
        / "scripts"
        / "logs"
        / "hooks.log"
    )
    tail = _bounded_log_tail(path, max_bytes=max_bytes)
    if not tail:
        return ()

    utc_now = now.astimezone(UTC)
    cutoff = utc_now - _ALERT_WINDOW
    redaction_env = _canonical_redaction_env()
    deduplicated: dict[
        tuple[datetime, str, str],
        tuple[datetime, str, str],
    ] = {}
    for line in tail.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            continue
        if not isinstance(record, dict):
            continue
        component = record.get("component")
        event = record.get("event")
        if (
            str(record.get("level", "")).casefold() != "error"
            or not isinstance(component, str)
            or not isinstance(event, str)
            or component not in _HOOK_COMPONENTS
            or event not in _HOOK_ERROR_EVENTS
        ):
            continue
        timestamp = _loaded_timestamp(record.get("timestamp"))
        if (
            timestamp is None
            or timestamp < cutoff
            or timestamp > utc_now + _FUTURE_SKEW
        ):
            continue
        message = _safe_message(record.get("message"), redaction_env)
        if message is None:
            continue
        key = (timestamp, component, message)
        deduplicated[key] = key

    ordered = sorted(
        deduplicated.values(),
        key=lambda value: (
            value[0],
            value[1],
            value[2],
        ),
        reverse=True,
    )[:_MAX_ALERTS]
    return tuple(
        HealthAlert(
            created_at=timestamp,
            level="error",
            message=message,
            component=component,
            redaction_env=redaction_env,
        )
        for timestamp, component, message in ordered
    )
