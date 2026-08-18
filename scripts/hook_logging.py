"""Fail-closed logging boundary shared by live hook entrypoints."""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

try:
    from .privacy import normalize_persistence_reason
    from .utils import open_secure_log_stream, prepare_secure_log_directory
except ImportError:  # Standalone execution with scripts/ on sys.path.
    from privacy import normalize_persistence_reason
    from utils import open_secure_log_stream, prepare_secure_log_directory


_HOOK_EVENTS = frozenset(
    {
        "hook_log",
        "malformed_input",
        "transcript_missing",
        "transcript_unreadable",
        "capture_failed",
        "queue_unavailable",
        "capture_succeeded",
        "capture_skipped",
    }
)
_SOURCE_AGENTS = frozenset({"claude", "codex"})
MAX_HOOK_CONTEXT_CHARS = 256


def _safe_context(value: object, env: dict[str, str]) -> str | None:
    if not isinstance(value, str) or not value or len(value) > MAX_HOOK_CONTEXT_CHARS:
        return None
    if value != value.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        return None
    if normalize_persistence_reason(value, env) != value:
        return None
    return value


def _safe_event_message(value: object, env: dict[str, str]) -> str:
    text = "".join(
        character if ord(character) >= 32 and ord(character) != 127 else " "
        for character in str(value)
    )
    return normalize_persistence_reason(text, env)


class HookJsonFormatter(logging.Formatter):
    """Format one stable, exception-safe operational JSONL record."""

    def __init__(self, component: str) -> None:
        super().__init__()
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC)
        event = getattr(record, "hook_event", "hook_log")
        if event not in _HOOK_EVENTS:
            event = "hook_log"
        message = record.getMessage()
        if event != "hook_log":
            message = _safe_event_message(message, dict(os.environ))
        value = {
            "timestamp": timestamp.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "level": record.levelname,
            "component": self.component,
            "event": event,
            "logger": record.name,
            "message": message,
        }
        source_agent = getattr(record, "source_agent", None)
        if source_agent in _SOURCE_AGENTS:
            value["source_agent"] = source_agent
        session_id = _safe_context(
            getattr(record, "session_id", None),
            dict(os.environ),
        )
        if session_id is not None:
            value["session_id"] = session_id
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class _HookLogHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        """Append one encoded JSON object with one operating-system write."""
        try:
            payload = (self.format(record) + "\n").encode("utf-8")
            written = os.write(self.stream.fileno(), payload)
            if written != len(payload):
                raise OSError("incomplete hook log append")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if self.stream is not None and not self.stream.closed:
                self.stream.close()
        finally:
            super().close()


def _remove_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_memory_hook_file", False):
            logger.removeHandler(handler)
            handler.close()


def _handler_log_path(handler: logging.Handler) -> Path | None:
    value = getattr(handler, "_memory_log_path", None)
    return Path(value) if isinstance(value, str) else None


def configure_hook_logger(
    logger_name: str,
    label: str,
    memory_root: Path | str,
) -> logging.Logger:
    """Configure one isolated hook logger, degrading to a null sink on risk."""
    logger = logging.getLogger(logger_name)
    target = Path(os.path.abspath(Path(memory_root).expanduser())) / "scripts" / "logs" / "hooks.log"
    tagged = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_memory_hook_file", False)
    ]
    if len(tagged) == 1 and _handler_log_path(tagged[0]) == target:
        return logger

    _remove_owned_handlers(logger)
    try:
        prepare_secure_log_directory(memory_root)
        handler: logging.Handler = _HookLogHandler(open_secure_log_stream(target))
    except (OSError, ValueError):
        handler = logging.NullHandler()
    handler.setFormatter(HookJsonFormatter(label))
    handler._memory_hook_file = True  # type: ignore[attr-defined]
    handler._memory_log_path = str(target)  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_hook_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: object,
    *,
    source_agent: str,
    session_id: object = None,
) -> None:
    """Emit one structured, bounded hook event without exception metadata."""
    logger.log(
        level,
        "%s",
        message,
        extra={
            "hook_event": event,
            "source_agent": source_agent,
            "session_id": session_id,
        },
    )


def classify_transcript_path(path: Path) -> str | None:
    """Classify a transcript path without reading or exposing its contents."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "transcript_missing"
    except OSError:
        return "transcript_unreadable"
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        return "transcript_unreadable"
    if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o444 == 0:
        return "transcript_unreadable"
    return None
